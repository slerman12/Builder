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

        self.frame_stack = frame_stack  # Can combine temporal image data via channel-stacking TODO Generalise to Time

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
    Step Agent in Env and return new experiences & eval/act logs
    """

    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        # Experiences to send to Replay or Logger, or directly to Agent-learn if offline streaming training
        experiences = self.step() if self.disable and self.on_policy and agent.training else []
        vlogs = []

        self.episode_done = self.disable

        step = frame = 0
        while not self.episode_done and step < steps:
            obs = self.exp.obs  # TODO Send whole exp to agent, or ordered datums depending on signature

            # Frame-stacked obs
            if hasattr(self.env, 'frame_stack'):
                if isinstance(self.exp.obs, torch.Tensor):
                    obs = obs.cpu().numpy()  # TODO Maybe standardize Env as Tensor

                obs = self.env.frame_stack(obs)

            obs = torch.as_tensor(obs, device=self.device)

            # Act
            store = Args()
            with act_mode(agent, self.ema):
                action = agent.act(obs, store)  # TODO Send in depending on signature, allow agent to output an exp

            # Inferred action will get passed to Env, Metrics, and Logger
            action = self.infer_action_from_action_spec(action)

            if not self.generate:
                action = action.cpu().numpy()  # TODO Maybe standardize Env as Tensor

            # Step Env, and add experiences to send to Replay (if agent training) or Logger (if agent evaluating)
            experiences += self.step(action, store)

            if vlog and hasattr(self.env, 'render') or self.generate:
                image_frame = action[:24].view(-1, *obs.shape[1:]) if self.generate \
                    else self.env.render()
                vlogs.append(image_frame)

            if agent.training:
                agent.step += 1
                agent.frame += len(action)

            step += 1
            frame += len(action)

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
    Take an Env step and return the experiences that need to be stored in Replay or logged.
    """
    def step(self, action=None, store=None):
        experiences = [self.exp]

        # The point of prev is to group data by time step since some datums, like reward, are delayed 1 time step
        prev, now = {}, {} if self.generate else self.env.step(action)  # Environment step

        # Parse, since prev is optional in Env
        if isinstance(now, tuple):
            prev, now = now
        prev, now = Args(prev or {}), Args(now or {})

        # Group all prev data
        if action is not None:
            self.exp.update(action=action)
        self.exp.update(prev)
        self.exp.update(store)

        # Episode done
        done = now.setdefault('done', True)

        self.exp.done = done

        # Transform
        if len(now.keys() - {'done', 'step'}):  # Unless empty
            now = self.transform(now, device=self.device)

        # Assume None action implies offline streaming training (metric, replay, logging not needed from Env)
        if action is not None:
            # Tally reward & logs
            self.tally_metric(self.exp)

            # Final "done" state should also be appended unless empty  TODO Replay should keep (prev, now) order
            if done and len(now.keys() - {'done', 'step'}):
                self.exp.done = False

                experiences.append(now)

        self.exp = now  # Becomes prev data

        # Done
        self.episode_done = done or self.episode_step > self.truncate_after - 2 or self.generate

        if done:
            self.exp = self.env.reset()
            self.exp = self.transform(self.exp, device=self.device)

        return experiences  # Send to Replay or Logger, or directly to Agent-learn if offline streaming training

    """
    Different environments have different inputs/outputs. 
    These methods infer or add default input (obs) and output (action) spec values, such as shape, action-space, & stats
    
    Default spec values
    - obs_spec: input shape and stats
    - action_spec: output shape, continuous vs. discrete, and stats
    """

    @cached_property
    def obs_spec(self):
        spec = Args({'shape': self.exp.obs.shape[1:] if 'obs' in self.exp else (),
                     **{'mean': None, 'stddev': None, 'low': None, 'high': None},
                     **getattr(self.env, 'obs_spec', {})})  # TODO Maybe override Env shape with self.exp shape
        #                                                       - (currently prioritizing Env spec)

        if self.frame_stack and int(self.frame_stack) > 1:
            assert spec['shape'], f'Frame stack {self.frame_stack} ' \
                                  f'specified but obs_shape has no channel-dims: {spec["shape"]}'

            # Frame stack
            spec['shape'] = torch.Size([spec['shape'][0] * self.frame_stack, *spec['shape'][1:]])

        # Update Env spec with defaults
        self.env.__dict__.setdefault('obs_spec', Args()).update(spec)

        return spec

    @cached_property
    def action_spec(self):
        spec = Args({**{'discrete_bins': None, 'low': None, 'high': None, 'discrete': False},
                     **getattr(self.env, 'action_spec', {})})

        # Infer discrete ranges
        if spec['discrete'] or spec['discrete_bins']:
            spec['discrete'] = True

            if spec['low'] is None:
                spec['low'] = 0
            if spec['high'] is None:
                spec['high'] = spec['discrete_bins'] - 1
            elif spec['discrete_bins'] is None:
                spec['discrete_bins'] = spec['high'] + 1

        # Infer action shape from label or action
        if 'shape' not in spec:
            if 'label' in self.exp:
                spec.shape = (len(self.exp.label),) if isinstance(self.exp.label, (tuple, set, list)) \
                    else (1,) if not hasattr(self.exp.label, 'shape') \
                    else (self.exp.label.shape[-1],) if spec['discrete'] else self.exp.label.shape
            elif 'action' in self.exp:
                spec.shape = self.exp.action.shape[1:]
            elif spec['discrete']:
                spec.shape = (1,)  # Discrete default is single-action

        # Update Env spec with defaults
        self.env.__dict__.setdefault('action_spec', Args()).update(spec)

        return spec

    """
    Metric tallying and tabulating for inference/evaluation
    """

    # Compute metric on each batch
    def tally_metric(self, exp):  # TODO Maybe standardize to Tensors
        # Convert batch to numpy
        for key, value in exp.items():
            if isinstance(value, torch.Tensor):
                exp[key] = value.cpu().numpy()

        # Compute each function-based (not string-based) metric on the batch. String-based gets computed in tabulate
        add = {key: m.add(exp) for key, m in self.metric.items() if callable(getattr(m, 'add', None))}

        # Add metrics to episode-long running list of metrics
        self.episode_adds.update({key: self.episode_adds.get(key, []) + [m]
                                  for key, m in add.items() if m is not None})

        # Always (try to) include reward especially if RL
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

    # Aggregate metrics over an episode
    def tabulate_metric(self):
        if self.episode_done:
            log = {key: self.metric[key].tabulate(episode)
                   for key, episode in self.episode_adds.items() if callable(getattr(self.metric.get(key, None),
                                                                                     'tabulate', None)) and episode}

            # Allow no return statement e.g. hacking metric for vlogging media
            log = {key: value for key, value in log.items() if value is not None}

            assert 'precision' in log, log  # TODO Why can log be empty??  python Run.py metric.precision=World.Metrics.Precision metric.recall=World.Metrics.Recall metric.F1='2*precision*recall/(precision+recall)'

            log.update({key: eval(m, None, log) for key, m in self.metric.items() if isinstance(m, str)})

            if 'reward' in self.episode_adds and 'reward' not in self.metric:
                # Reward, by default, sums
                log['reward'] = sum(self.episode_adds['reward'])

            return log

        return {}

    """
    Action can be inferred from action_spec, even if the agent's output action doesn't perfectly match
    as expected in action_spec, e.g., in discrete Envs, multi-dim actions can be inferred as logits/probas and argmax'd 
    when action_spec expects shape (1,). Action also may get broadcast to expected shape.
    """
    def infer_action_from_action_spec(self, action):
        shape = self.action_spec['shape']

        try:
            # Broadcast to expected shape
            action = action.reshape(len(action), *shape)  # Assumes a batch dim
        except (ValueError, RuntimeError) as e:
            # Arg-maxes if categorical distribution passed in
            if self.action_spec['discrete']:
                try:
                    action = action.reshape(len(action), -1, *shape)  # Assumes a batch dim
                except:
                    raise RuntimeError(f'Discrete environment could not broadcast or adapt action of shape '
                                       f'{action.shape} to expected batch-action shape {(-1, *shape)}')
                action = action.argmax(1)
            else:
                raise e

        discrete_bins, low, high = self.action_spec['discrete_bins'], self.action_spec['low'], self.action_spec['high']

        # Round to nearest decimal/int corresponding to discrete bins, high, and low
        if self.action_spec['discrete']:
            action = torch.round((action - low) / (high - low) * (discrete_bins - 1)) / \
                     (discrete_bins - 1) * (high - low) + low
        else:
            # TODO Generalise to regression
            pass

        return action


"""
"Act mode": No gradients, no randomness like Dropout, and exponential moving average (EMA) if toggled 
"""


# Toggles / resets eval, inference, and EMA modes
class act_mode:
    def __init__(self, agent, ema=False):
        self.agent = agent

        self.models = {key: getattr(agent, key) for key in {'encoder', 'actor'} if hasattr(agent, key)}

        # TODO Assuming blocks. Eval should be entered agent-wise ? iterated .children() ?
        #   A.K.A. Generalise to Agents/Models that don't use Encoder/Actor Blocks
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
