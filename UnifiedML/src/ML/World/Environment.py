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
        experiences = [*([self.env.step(None)] if self.disable and self.on_policy and agent.training else [])]
        vlogs = []

        self.episode_done = self.disable

        step = frame = 0
        while not self.episode_done and step < steps:
            exp = self.exp

            # Frame-stacked obs
            obs = getattr(self.env, 'frame_stack', lambda x: x)(exp.obs)  # TODO Send whole exp to agent
            obs = torch.as_tensor(obs, device=self.device)

            # Act
            store = Args()
            with act_mode(agent, self.ema):
                action = agent.act(obs, store)  # TODO Allow agent to output an exp

            # Inferred action will get passed to Env, Metrics, and Logger. However, original will be stored in Replay.
            # TODO No need. store accounts for this
            _action = self.infer_action_from_action_spec(action)

            if not self.generate:
                _action = _action.cpu().numpy()  # TODO Maybe standardize Env as Tensor

            prev, now = {}, {} if self.generate else self.env.step(_action)  # Experience

            # The point of prev is to group data by time step since some datums, like reward, are delayed 1 time step
            if isinstance(now, tuple):
                prev, now = now

            now = now or {}  # Can be None

            self.exp.update(store)
            self.exp.update(action=now.get('action', _action), step=agent.step)
            self.exp.update(prev)  # Prev is only needed for metric

            # Tally reward & logs
            self.tally_metric(self.exp)

            if 'reward' not in now and 'reward' in self.exp:
                # Note: Envs should either always return a reward or never return a reward.
                # Inconsistent presence of reward can lead to tally_metric carrying over last-present
                # reward to next time steps
                now.reward = self.exp.reward  # Metric can actually set reward, e.g., if RL

            now.action = action

            # Transform
            now = self.transform(now)

            # TODO Problem: online (and offline streaming at point 5):
            #  (1) skips storing first batch in Replay,
            #         I would say do: experiences.append(Args({'step': agent.step, **self.exp, **prev, **now, **store}))
            #         but self.exp is updated by Metrics... not needed for Replay.
            #         Make/return copy in tally_metric? Or, just before tally_metric: exp = Args(self.exp)
            #         Then: experiences.append(Args({'step': agent.step, **exp, **prev, **now, **store}))
            #         But how does that work for RL? The first batch still doesn't have consistent corresponding datums
            #         ! Maybe Replay should just pair exp and prev and now should never explicitly be stored
            #         Then prev is necessary for RL to pair the transitions correctly
            #  (2) stores only "step" and "done" and "action" in Reply at last batch for Datums
            #       Don't append anything in that case (if now.keys() union prev.keys() \subseteq {'done', 'action'})?
            #  (3) "now.action = action" means action always 1-time-step delayed. Does Replay always account for that?
            #         This wouldn't be an issue if experiences stored (step, exp, prev, store) for Replay and not now
            #         Wait - but action should be prev, not now for RL but now for Datums? Why? Why not now for both?
            #           Well, the first batch wouldn't have a corresponding action. After that, not sure...
            #           Why not prev for both? I think do: "prev.action = action" instead. No time-step delay.
            #           Or nothing, since already added to self.exp/exp
            #  (4) Don't forget to update the second-to-last experience/transition stored in Replay and metric'd and
            #      logged as done!
            #  (5) How would offline streaming respond to this? exp, prev need to be unified; done added to 2nd-to-last;
            #      & unified (exp, prev, with potentially done) as output; what about metric-originated reward, ignore?.

            if agent.training:
                # These go to Replay and include now (mixed time steps)
                experiences.append(Args({'step': agent.step, **prev, **now, **store}))
            else:
                # These go to logger and don't include now
                experiences.append(self.exp)

            self.exp = now
            if 'obs' in self.exp and isinstance(self.exp.obs, torch.Tensor) and hasattr(self.env, 'frame_stack'):
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

            done = self.exp.get('done', True)

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
                     **getattr(self.env, 'obs_spec', {})})

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

        # Update Env spec with defaults
        self.env.__dict__.setdefault('action_spec', Args()).update(spec)

        return spec

    """
    Metric tallying and tabulating for inference/evaluation
    """
    # Compute metric on batch
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

        # Always include reward
        if 'reward' in self.metric and 'reward' in self.episode_adds:
            exp.reward = self.episode_adds['reward'][-1]  # Override Env reward with metric reward
        elif ('reward' in self.metric or 'reward' in exp) and not getattr(self, 'popped_reward', None):
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

            # To avoid this becoming the reward used next time in the previous elif block, and then never changing
            setattr(self, 'popped_reward', True)

    # Aggregate metrics
    def tabulate_metric(self):
        if self.episode_done:
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
    Action can be inferred from action_spec, even if the agent's output action doesn't perfectly match
    action_spec, e.g., in discrete Envs, multi-dim actions can be inferred as logits/probas and argmax'd 
    when action_spec expects shape (1,). Action also gets broadcast to expected shape.
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
            # TODO Generalize to regression
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
