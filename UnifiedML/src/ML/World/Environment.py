# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
import warnings
from math import inf

import torch

from Utils import Transform

from minihydra import instantiate, get_module, valid_path, Args


class Environment:
    def __init__(self, env, frame_stack=1, truncate_episode_steps=1e3, action_repeat=1, RL=True,
                 offline=False, stream=True, generate=False, ema=False, train=True, seed=0,
                 transform=None, device='cpu', obs_spec=None, action_spec=None):
        self.RL = RL
        self.offline = offline
        self.generate = generate

        self.device = device
        self.transform = Transform(instantiate(env.pop('transform', transform), device=device))

        # Offline and generate don't use training rollouts! Unless on-policy (stream)
        self.disable, self.on_policy = (offline or generate) and train, stream

        self.truncate_after = train and truncate_episode_steps or inf  # Truncate episodes shorter (inf if None)

        if not self.disable or stream:
            self.env = instantiate(env, frame_stack=int(stream) or frame_stack, action_repeat=action_repeat, RL=RL,
                                   offline=offline, generate=generate, train=train, seed=seed, device=device)
            # Experience
            self.exp = self.env.reset()
            self.exp = self.transform(self.exp, device=self.device)

            self.obs_spec = Args({**{'mean': None, 'stddev': None, 'low': None, 'high': None},
                                  **getattr(self.env, 'obs_spec', {}), **(obs_spec or {})})
            self.action_spec = Args({**{'discrete_bins': None, 'low': None, 'high': None, 'discrete': False},
                                     **getattr(self.env, 'action_spec', {}), **(action_spec or {})})

        self.action_repeat = getattr(getattr(self, 'env', 1), 'action_repeat', 1)  # Optional, can skip frames

        self.episode_sums = {}

        self.metric = {key: get_module(metric) if valid_path(metric) else metric
                       for key, metric in env.metric.items() if metric is not None}
        self.ema = ema  # Can use an exponential moving average model

        self.episode_done = self.episode_step = self.episode_frame = self.last_episode_len = 0
        self.daybreak = None

    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        experiences = [*([self.env.step()] if self.disable and self.on_policy else [])]
        vlogs = []

        self.episode_done = self.disable

        step = frame = 0
        while not self.episode_done and step < steps:
            exp = self.exp

            # Frame-stacked obs
            obs = getattr(self.env, 'frame_stack', lambda x: x)(exp.obs)

            # Act
            store = Args()
            with act_mode(agent, self.ema):
                action = agent.act(obs, store)

            exp = Args(action=action) if self.generate else self.env.step(action.cpu().numpy())  # Experience

            # Transform
            exp = self.transform(exp, device=self.device)

            # Tally reward & logs
            self.tally_metric(exp)

            exp.update(**store, step=agent.step)
            experiences.append(exp)

            if vlog and hasattr(self.env, 'render') or self.generate:
                image_frame = action[:24].view(-1, *exp.obs.shape[1:]) if self.generate \
                    else self.env.render()
                vlogs.append(image_frame)

            if agent.training:
                agent.step += 1
                agent.frame += len(action)

            step += 1
            frame += len(action)
            
            done = exp.get('done', True)

            # Done
            self.episode_done = done or self.episode_step > self.truncate_after - 2 or self.generate

            if done:
                self.exp = self.env.reset()
                self.exp = self.transform(self.exp, device=self.device)

        agent.episode += agent.training * self.episode_done  # Increment agent episode

        # Tally time
        self.episode_step += step
        self.episode_frame += frame

        if self.episode_done:
            self.last_episode_len = self.episode_step

        # Log stats
        sundown = time.time()
        frames = self.episode_frame * self.action_repeat

        log = {'time': sundown - agent.birthday,
               'step': agent.step,
               'frame': agent.frame * self.action_repeat,
               'epoch' if self.offline or self.generate else 'episode':
                   (self.offline or self.generate) and agent.epoch or agent.episode, **self.episode_sums,
               'fps': frames / (sundown - self.daybreak)} if not self.disable \
            else None

        if self.episode_done:
            self.episode_sums = {}
            self.episode_step = self.episode_frame = 0
            self.daybreak = sundown

        return experiences, log, vlogs

    def tally_metric(self, exp):
        metric = {key: m(exp) for key, m in self.metric.items() if callable(m)}
        if 'reward' in exp and 'reward' not in self.metric:
            metric['reward'] = exp.reward.mean() if hasattr(exp.reward, 'mean') else exp.reward
        metric.update({key: eval(m, None, metric) for key, m in self.metric.items() if isinstance(m, str)})
        exp.update({key: m for key, m in metric.items() if key in exp or key == 'reward'})

        # Use random popped metric as reward if RL and 'reward' not in exp
        if 'reward' not in exp and len(metric) and self.RL:  # Note: AC2 classifier actually defines its own reward
            key = next(iter(metric.keys()))
            exp.reward = metric['reward'] = metric.pop(key)
            warnings.warn(f'"RL" enabled but no Env reward found. Using metric "{key}" as reward. '
                          f'Customize your own reward with the "reward=" flag. For example: '
                          f'"reward=path.to.rewardfunction" or even "reward=-{key}+1". See docs for more demos.')

        self.episode_sums.update({key: self.episode_sums[key] + m if key in self.episode_sums else m
                                  for key, m in metric.items()})


# Temporarily switches on eval() mode for specified blocks; then resets them.
# Enters and exits Pytorch inference mode
# Enables / disables EMA
class act_mode:
    def __init__(self, agent, ema=False):
        self.agent = agent

        self.models = {key: getattr(agent, key) for key in {'encoder', 'actor'} if hasattr(agent, key)}
        self.inference = torch.inference_mode()

        self.ema = ema

    def __enter__(self):
        # Disables things like dropout, etc.
        self.mode_model = [(model.training, model.eval()) for model in self.models.values()]

        self.inference.__enter__()

        # Exponential moving average (EMA)
        if self.ema:
            [setattr(self.agent, key, model.ema) for key, model in self.models.items()]

    def __exit__(self, *args, **kwargs):
        [model.train(mode) for mode, model in self.mode_model]
        self.inference.__exit__(*args, **kwargs)

        if self.ema:
            [setattr(self.agent, key, model) for key, model in self.models.items()]
