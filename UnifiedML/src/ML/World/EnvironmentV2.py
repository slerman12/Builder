# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
import warnings
from functools import cached_property
from math import inf

import torch
from minihydra import instantiate, valid_path, Args

from Utils import Modals


# TODO (See red journal first page for flowchart diagram of current code)


class Environment:
    def __init__(self, env, frame_stack=1, truncate_episode_steps=1e3, action_repeat=1, RL=True, offline=False,
                 stream=True, generate=False, ema=False, train=True, seed=0, transform=None, device='cpu',
                 obs_spec=None, action_spec=None):
        self.RL = RL
        self.offline = offline
        self.generate = generate

        self.device = device
        # Support env.transform= or environment.transform= and with obs or exp as transform input
        self.transform = Modals(instantiate(env.pop('transform', transform), device=device))

        # Offline and generate don't use training rollouts! Unless on-policy (stream)
        self.disable, self.on_policy = (offline or generate) and train, stream

        self.truncate_after = train and truncate_episode_steps or inf  # Truncate episodes shorter (inf if None)

        if not self.disable or stream:
            self.env = instantiate(env, frame_stack=int(stream) or frame_stack, action_repeat=action_repeat, RL=RL,
                                   offline=offline, generate=generate, train=train, seed=seed, device=device)
            # Experience
            self.exp = self.transform(self.env.reset(), device=self.device)

            # Allow overriding specs
            self.obs_spec.update(obs_spec)
            self.action_spec.update(action_spec)

            # Default action space is continuous (not discrete) if unspecified
            if self.action_spec.discrete == '???':
                self.action_spec.discrete = getattr(self.env, 'action_spec', {}).get('discrete', False)

        self.action_repeat = getattr(getattr(self, 'env', 1), 'action_repeat', 1)  # Optional, can skip frames

        self.episode_adds = {}
        self.metric = {key: instantiate(metric) if valid_path(metric) else metric
                       for key, metric in env.metric.items() if metric is not None}

        self.ema = ema  # Can use an exponential moving average model

        self.episode_done = self.episode_step = self.episode_frame = self.last_episode_len = 0
        self.daybreak = None

    """
    Step the agent in the environment and return new experiences and inference/acting logs
    """
    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        # If offline streaming (no Replay) and training, then take an Env step (get next batch) without an Agent action
        experiences = [*([self.env.step()] if self.disable and self.on_policy and agent.training else [])]
        vlogs = []

        self.episode_done = self.disable

        step = frame = 0
        while not self.episode_done and step < steps:
            exp = self.exp

            # Frame-stacked obs
            obs = getattr(self.env, 'frame_stack', lambda x: x)(exp.obs)  # TODO Send whole exp to agent
            obs = torch.as_tensor(obs, device=self.device)

            # Act
            store = Args()  # TODO Pass in log as well - problem, has to read var name order (custom name order weird)
            with act_mode(agent, self.ema):
                action = agent.act(obs, store)  # TODO Allow agent to output an exp

            if not self.generate:
                action = action.cpu().numpy()  # TODO Maybe standardize Env as Tensor

            prev, now = {}, {} if self.generate else self.env.step(action)  # Experience TODO Just changed this
            # prev = now = {} if self.generate else self.env.step(action)  # Experience

            # The point of prev is to group data by time step for intuitive metrics of corresponding datums
            # (e.g. corresponding action and obs)
            if isinstance(now, tuple):
                prev, now = now

            now = now or {}  # Can be None

            # TODO Metric only logged for previous.
            #  Doesn't get logged for last, and definitely not for done, (unless prev = now)
            self.exp.update(store)
            self.exp.update(action=now.get('action', action), step=agent.step)
            self.exp.update(prev)  # Prev is only needed for metric

            # Tally reward & logs
            self.tally_metric(self.exp)  # TODO Metric only consists of action if prev is missing.
            #                                   "now" label doesn't even make it? - it does, via self.exp persistence
            #                                   - Just fixed this via "prev = now =" instead of "prev, now = {}, {} if"
            #                                   Or did it carry over from the previous self.exp? - yes
            #                                   Yes, action was/is one time step delayed in Datums
            #                                       - Doesn't have to be? Or should be in "prev"?

            if 'reward' not in now and 'reward' in self.exp:
                now.reward = self.exp.reward  # Metric can actually set reward, e.g. if RL

            # if agent.training:
            #     experiences.append(self.exp)  # TODO Replay expects prev and exp together. I think can delete this

            # TODO I don't remember what this is
            # now.update({'prev_' + key: value for key, value in self.exp.items() if key not in now})  # TODO -Transform

            # Transform
            now = self.transform(now)

            if agent.training:
                # These go to Replay and include now (mixed time steps)
                experiences.append(Args({'action': action, 'step': agent.step, **prev, **now, **store}))
            else:
                # These go to logger and don't include now  TODO: Is this wrong? Should just use above for both cases?
                experiences.append(self.exp)                # TODO I think so. preditced_vs_actual needs "now", right?
            #                                       # TODO Depends. If action is prev and corresponds to prev label,
            #                                           then depends what "done"-state corresponds to
            # TODO Trying this instead: (Perhaps this above is the bug causing the latest batch to not be logged)
            #   (But now wouldn't the first batch end up not logged?)
            # experiences.append(Args({'action': action, 'step': agent.step, **prev, **now, **store}))

            self.exp = now
            if isinstance(self.exp.obs, torch.Tensor) and hasattr(self.env, 'frame_stack'):
                # TODO What if not Numpy? Assume frame stack Tensor? Transform as class?
                self.exp.obs = self.exp.obs.numpy()

            if vlog and hasattr(self.env, 'render') or self.generate:
                image_frame = action[:24].view(-1, *self.exp.obs.shape[1:]) if self.generate \
                    else self.env.render()
                vlogs.append(image_frame)

            if agent.training:
                agent.step += 1
                agent.frame += len(action)

            step += 1
            frame += len(action)

            # TODO Perhaps this is the bug. See TODO of line 166 of Datums - reprogram/clean Datums!
            done = self.exp.get('done', True)  # TODO Datums skips last batch (self.exp doesn't get acted on) (maybe nstep controls this)

            # Done
            self.episode_done = done or self.episode_step > self.truncate_after - 2 or self.generate

            if done:
                # TODO Last "now" doesn't get acted on, metric'd, or logged (though does get sent to replay)
                #     This is correct for RL, but not for supervised where
                #     all of that is still necessary but w/o "prev" or env-step. Can't do this for RL
                #     since no corresponding reward for metric or log. Maybe for supervised,
                #     done "now" can be: (now or {})  [and in Datums, change the "done" output to None.
                #                                       Or "now" becomes {} here if "done"? Or both]
                #     and in Datums reset and sample can be more separated, with done state being None or {'done': True}
                #   I think this summarizes the whole bug and all that's left.
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

        log = self.tabulate_metric()

        log = {'time': sundown - agent.birthday,
               'step': agent.step,
               'frame': agent.frame * self.action_repeat,
               'epoch' if self.offline or self.generate else 'episode':
                   (self.offline or self.generate) and agent.epoch or agent.episode, **log,
               'fps': frames / (sundown - self.daybreak)} if not self.disable \
            else None

        if self.episode_done:
            self.episode_adds = {}
            self.episode_step = self.episode_frame = 0
            self.daybreak = sundown

        return experiences, log, vlogs

    """
    Different environments have different inputs/outputs
    
    Default spec values
    - obs_spec: input shape and stats
    - action_spec: output shape, continuous vs. discrete, and stats
    """
    @cached_property
    def obs_spec(self):
        spec = Args({'shape': self.exp.obs.shape[1:] if 'obs' in self.exp else (),
                     **{'mean': None, 'stddev': None, 'low': None, 'high': None},
                     **getattr(self.env, 'obs_spec', {})})

        self.env.__dict__.setdefault(self.env, 'obs_spec', {})['obs_spec'] = spec

        return spec

    @cached_property
    def action_spec(self):
        spec = Args({**{'discrete_bins': None, 'low': None, 'high': None, 'discrete': False},
                     **getattr(self.env, 'action_spec', {})})

        if 'shape' not in spec:
            # Infer action shape from label or action
            if 'label' in self.exp:
                spec.shape = (len(self.exp.label)) if isinstance(self.exp.label, (tuple, set, list)) \
                    else (1,) if not hasattr(self.exp.label, 'shape') \
                    else len(self.exp.label.shape[-1]) if spec['discrete'] else self.exp.label.shape
            elif 'action' in self.exp:
                spec.shape = self.exp.action.shape[1:]

        self.env.__dict__.setdefault(self.env, 'action_spec', {})['action_spec'] = spec

        return spec

    """
    Metric tallying and tabulating for inference/evaluation
    """
    # Compute metric on batch
    def tally_metric(self, exp):
        # TODO Maybe standardize to Tensors
        for key, value in exp.items():
            if isinstance(value, torch.Tensor):
                exp[key] = value.cpu().numpy()

        add = {key: m.add(exp) for key, m in self.metric.items() if callable(getattr(m, 'add', None))}

        self.episode_adds.update({key: self.episode_adds.get(key, []) + [m]
                                  for key, m in add.items() if m is not None})

        # Always include reward
        if 'reward' in self.metric and 'reward' in self.episode_adds:
            exp.reward = self.episode_adds['reward'][-1]  # Override Env reward with metric reward
        elif 'reward' in self.metric or 'reward' in exp:
            # Evaluate reward from string  Note: No recursion between strings
            if isinstance(self.metric.get('reward', None), str):
                exp.reward = eval(self.metric['reward'], {key: episode[-1]
                                                          for key, episode in self.episode_adds.items()})
            # Override metric reward with Env reward
            self.episode_adds.setdefault('reward', []).append(exp.reward.mean() if hasattr(exp.reward, 'mean')
                                                              else exp.reward)
        elif len(self.episode_adds) and self.RL:
            # Use random popped metric as reward if RL
            key = next(iter(self.episode_adds.keys()))
            self.episode_adds.setdefault('reward', []).append(self.episode_adds.pop(key))
            exp.reward = self.episode_adds['reward'][-1]
            warnings.warn(f'"RL" enabled but no Env reward found. Using metric "{key}" as reward. '
                          f'Customize your own reward with the "reward=" flag. For example: '
                          f'"reward=path.to.rewardfunction" or even "reward=-{key}+1". See docs for more demos.')

    # Aggregate metrics
    def tabulate_metric(self):
        if self.episode_done:
            # TODO See here: episode is inexplicably adding a batch at 0th step???
            # log = {key: (self.metric[key].tabulate(episode), print(len(episode)))[0]
            #        for key, episode in self.episode_adds.items() if callable(getattr(self.metric.get(key, None),
            #                                                                          'tabulate', None)) and episode}

            log = {key: self.metric[key].tabulate(episode)
                   for key, episode in self.episode_adds.items() if callable(getattr(self.metric.get(key, None),
                                                                                     'tabulate', None)) and episode}

            # Allow no return statement e.g. hacking metric for vlogging media
            log = {key: value for key, value in log.items() if value is not None}

            log.update({key: eval(m, None, log) for key, m in self.metric.items() if isinstance(m, str)})

            if 'reward' in self.episode_adds and 'reward' not in self.metric:
                # Reward, by default, sums
                log['reward'] = sum(self.episode_adds['reward'])

            return log

        return {}


"""
    "Act mode": No gradients, no randomness like Dropout, and exponential moving average (EMA) if toggled 
"""


# Toggles / resets eval, inference, and EMA modes
class act_mode:
    def __init__(self, agent, ema=False):
        self.agent = agent

        self.models = {key: getattr(agent, key) for key in {'encoder', 'actor'} if hasattr(agent, key)}

        # TODO Assuming blocks. Eval should be entered agent-wise ? iterated .children() ?
        # if ema and not self.models:
        #     # TODO ema_decay each time learn called
        #     # TODO ema_begin_step (no point early in training in supervised setting - RL act/eval-ema matters)
        #     self.ema = agent.__dict__.setdefault('_ema_', deepcopy(agent).requires_grad_(False).eval()).act
        #     self.models['act'] = self

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

    # def __call__(self, obs, store):
    #     return self.agent.act(obs, store)
