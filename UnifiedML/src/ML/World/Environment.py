# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
import warnings
from functools import cached_property
from math import inf

import torch
from minihydra import instantiate, get_module, valid_path, Args

from Utils import Transform


class Environment:
    def __init__(self, env, frame_stack=1, truncate_episode_steps=1e3, action_repeat=1, RL=True, offline=False,
                 stream=True, generate=False, ema=False, train=True, seed=0, transform=None, device='cpu',
                 obs_spec=None, action_spec=None):
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
            exp = self.env.reset()
            self.exp = self.transform(exp, device=self.device)

            # Update
            self.obs_spec.update(obs_spec)
            self.action_spec.update(action_spec)

        self.action_repeat = getattr(getattr(self, 'env', 1), 'action_repeat', 1)  # Optional, can skip frames

        self.episode_adds = {}
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
            obs = torch.as_tensor(obs, device=self.device)

            # Act
            store = Args()
            with act_mode(agent, self.ema):
                action = agent.act(obs, store)

            exp = Args(action=action) if self.generate else self.env.step(action.cpu().numpy())  # Experience

            # Transform
            exp = self.transform(exp)

            # Tally reward & logs
            self.tally_metric(exp)

            exp.update(**store, step=agent.step)
            experiences.append(exp)

            self.exp = Args(exp)
            if isinstance(exp.obs, torch.Tensor) and hasattr(self.env, 'frame_stack'):
                self.exp.obs = exp.obs.numpy()

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
                   (self.offline or self.generate) and agent.epoch or agent.episode, **self.episode_adds,
               'fps': frames / (sundown - self.daybreak)} if not self.disable \
            else None

        if self.episode_done:
            self.episode_adds = {}
            self.episode_step = self.episode_frame = 0
            self.daybreak = sundown

        return experiences, log, vlogs

    @cached_property
    def obs_spec(self):
        return Args({'shape': self.exp.obs.shape if 'obs' in self.exp else (),
                     **{'mean': None, 'stddev': None, 'low': None, 'high': None},
                     **getattr(self.env, 'obs_spec', {})})

    @cached_property
    def action_spec(self):
        spec = {**{'discrete_bins': None, 'low': None, 'high': None, 'discrete': False},
                **getattr(self.env, 'action_spec', {})}

        if 'shape' not in spec:
            # Infer action shape from label or action
            if 'label' in self.exp:
                spec.shape = (len(self.exp.label)) if isinstance(self.exp.label, (tuple, set, list)) \
                    else (1,) if not hasattr(self.exp.label, 'shape') \
                    else len(self.exp.label.shape[..., -1]) if spec['discrete'] else self.exp.label.shape[..., -1]
            elif 'action' in self.exp:
                spec.shape = self.exp.action.shape

        return Args(spec)

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

        self.episode_adds.update({key: self.episode_adds[key] + m if key in self.episode_adds else m
                                  for key, m in metric.items()})


# Toggles / resets eval, inference, and EMA modes
class act_mode:
    def __init__(self, agent, ema=False):
        self.agent = agent

        self.models = {key: getattr(agent, key) for key in {'encoder', 'actor'} if hasattr(agent, key)}
        self.inference = torch.inference_mode()

        self.ema = ema

    def __enter__(self):
        # Disables randomness, dropout, etc.
        self.mode_model = [(model.training, model.eval()) for model in self.models.values()]

        self.inference.__enter__()  # Disables gradients

        # Exponential moving average (EMA)
        if self.ema:
            [setattr(self.agent, key, model.ema) for key, model in self.models.items()]

    def __exit__(self, *args, **kwargs):
        [model.train(mode) for mode, model in self.mode_model]
        self.inference.__exit__(*args, **kwargs)

        if self.ema:
            [setattr(self.agent, key, model) for key, model in self.models.items()]
