# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
from math import inf

import torch
from minihydra import instantiate, valid_path


class Environment:
    def __init__(self, env, suite='DMC', task='cheetah_run', frame_stack=1, truncate_episode_steps=1e3, action_repeat=1,
                 offline=False, stream=False, generate=False, train=True, seed=0, transform=None, device='cpu'):
        self.suite = suite.lower()
        self.offline = offline
        self.generate = generate

        self.device = device
        self.transform = instantiate(env.pop('transform', transform)) or (lambda _: _)

        # Offline and generate don't use training rollouts! Unless on-policy (stream)
        self.disable, self.on_policy = (offline or generate) and train, stream

        self.truncate_after = train and truncate_episode_steps or inf  # Truncate episodes shorter (inf if None)

        if not self.disable or stream:
            self.env = instantiate(env, task=task, frame_stack=int(stream) or frame_stack, action_repeat=action_repeat,
                                   offline=offline, generate=generate, train=train, seed=seed, device=device)
            self.env.reset()

        self.action_repeat = getattr(getattr(self, 'env', 1), 'action_repeat', 1)  # Optional, can skip frames

        self.episode_sums = {}
        self.metric = {key: instantiate(metric) if valid_path(metric) else metric
                       for key, metric in env.metric.items() if metric is not None}

        self.episode_done = self.episode_step = self.episode_frame = self.last_episode_len = 0
        self.daybreak = None

    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        experiences = [*([self.env.step()] if self.disable and self.on_policy else [])]
        video_image = []

        self.episode_done = self.disable

        step = frame = 0
        while not self.episode_done and step < steps:
            exp = self.env.exp

            # Frame-stacked obs
            obs = getattr(self.env, 'frame_stack', lambda x: x)(exp.obs)
            obs = self.transform(torch.as_tensor(obs, device=self.device))

            # Act
            with torch.no_grad():
                action, store = agent.act(obs)

            if not self.generate:
                exp = self.env.step(action.cpu().numpy())  # Experience

            # Tally reward & logs
            self.tally_metric(exp)

            exp.update(**store, step=agent.step)
            experiences.append(exp)

            if vlog or self.generate:
                image_frame = action[:24].view(-1, *exp.obs.shape[1:]) if self.generate \
                    else self.env.render()
                video_image.append(image_frame)

            step += 1
            frame += len(action)

            # Done
            self.episode_done = self.env.episode_done or self.episode_step > self.truncate_after - 2 or self.generate

            if self.env.episode_done:
                self.env.reset()

        agent.episode += agent.training * self.episode_done  # Increment agent episode

        # Tally time
        self.episode_step += step
        self.episode_frame += frame

        if self.episode_done:
            self.last_episode_len = self.episode_step

        # Log stats
        sundown = time.time()
        frames = self.episode_frame * self.action_repeat

        logs = {'time': sundown - agent.birthday,
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

        return experiences, logs, video_image

    def tally_metric(self, exp):
        metric = {key: m(exp) for key, m in self.metric.items() if callable(m)}
        if 'reward' in exp and 'reward' not in self.metric:
            metric['reward'] = exp.reward.mean() if any(k in str(type(exp.reward)) for k in {'np', 'torch'}) \
                else exp.reward
        metric.update({key: eval(m, None, metric) for key, m in self.metric.items() if isinstance(m, str)})
        exp.update({key: m for key, m in metric.items() if key in exp or key == 'reward'})

        self.episode_sums.update({key: self.episode_sums[key] + m if key in self.episode_sums else m
                                  for key, m in metric.items()})
