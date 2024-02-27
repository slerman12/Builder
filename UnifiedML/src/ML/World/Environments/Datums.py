# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import random
import os
import resource
import warnings

import numpy as np
import torch

from torch.utils.data import DataLoader

from World.Dataset import load_dataset, worker_init_fn, compute_stats
from minihydra import Args


class Datums:
    """
    Envs must:
    - have a step() function
    - have a reset() function

        The step() and reset() functions output "experiences"
            i.e. dicts containing datums like "obs", "action", "reward", 'label", "done"

        The end of an episode or epoch can be signaled by the "done" boolean. It will be assumed True if excluded.
        This can be useful for organizing temporal data or tabulating metrics based on multiple steps of batches.
        - The "done" step doesn't get acted on. Its contents can be a no-op (either output None or just {'done': True}),
        or can include any additional datums needed for logging or storing in Replay, such as "reward".
        - The reset() state can't ever be a "done" state even if you specify it as such. In that case,
        output None from your step() function.

    Envs can:
    - have an obs_spec dict
    - have an action_spec dict

        depending on what can be inferred. For example, obs_spec.shape often isn't necessary
        since it can be inferred from the "obs" output of the reset() function.

        For obs_spec, these stats can include: "shape", "mean", "stddev", "low", "high".
        Useful for standardization and normalization.

        For action_spec, these stats can include: "shape", "discrete_bins", "low", "high", "discrete".
        For discrete action spaces and action normalization.

    ---

    Datasets must:
    - extend Pytorch Datasets
    - output (obs, label) pairs, or dicts of named datums, e.g., {'obs': obs, 'label': label, ...}

    Datasets can:
    - include a "classes" attribute that lists the different class names or classes
        This allows quick and exact computation of number of classes being predicted from, without having to count
        via iterating through the whole dataset.

    ---

    The "step" function has a no-op default action (action=None) in its signature to allow for streaming (from Env
    rather than Replay) in Offline-mode since Offline datasets, even in an Env, don't require actions to sample the
    next batch, unlike RL. But in case of Online training, where a history of actions do get stored in Replay,
    logits here stored as argmax.
    """
    def __init__(self, dataset, test_dataset=None, train=True, offline=True, generate=False, batch_size=8,
                 num_workers=1, standardize=False, norm=False, device='cpu', **kwargs):
        if not train and test_dataset is not None:
            # Inherit from test_dataset
            dataset = test_dataset

        # (This might not generally be necessary)
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)  # Shared memory can create a lot of file descriptors
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))  # Increase soft limit to hard limit

        dataset = load_dataset('World/ReplayBuffer/Offline/', dataset, allow_memory=False, train=train)

        self.discrete = hasattr(dataset, 'classes')  # load_dataset adds a .classes attribute to the dataset if discrete

        # CPU workers
        num_workers = max(1, min(num_workers, os.cpu_count()))

        self.batches = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=torch.device(device).type == 'cuda',
                                  collate_fn=getattr(dataset, 'collate_fn', None),  # Useful if streaming dynamic lens
                                  worker_init_fn=worker_init_fn)  # TODO Support using Replay in Env (and by default)

        self.num_sampled_batches = 0

        self._batches = iter(self.batches)

        # Fill in necessary obs_spec and action_spec stats (e.g. mean, stddev, low, high) from dataset
        if train and (offline or generate) and (standardize or norm):
            # TODO Alternatively, load_dataset can output Args of recollected stats as well;
            #  maybe specify what to save in card replay
            self.obs_spec = compute_stats(self.batches)

        self.action_spec = Args({'discrete_bins': len(dataset.classes) if self.discrete else None})

        # Experience
        self.exp = None

    def step(self, action=None):
        if self.num_sampled_batches == len(self):  # Episode is "done" at end of epoch
            return dict(done=True)

        self.reset()  # Re-sample datums for self.exp

        if action is not None:  # No action - "no-op" - allowed for Offline streaming
            # Adapt to discrete!    Note: storing argmax
            self.exp.action = self.adapt_to_discrete(action) if self.discrete \
                else np.reshape(action, self.exp.label.shape) if 'label' in self.exp \
                else action  # TODO Does the Agent already do this argmax? Double-check if this code block is necessary

        return self.exp

    def reset(self):  # The reset step is never stored
        batch = self.sample()

        self.num_sampled_batches += 1

        # TODO Should provide per-Datum specs
        self.exp = Args(**batch)

        for key, value in self.exp.items():
            self.exp[key] = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.array(value)

        if 'label' in self.exp:
            if len(self.exp.label.shape) == 1:
                self.exp.label = np.expand_dims(self.exp.label, 1)

        self.exp.done = False

        return self.exp

    def sample(self):
        try:
            batch = next(self._batches)
        except StopIteration:
            self.num_sampled_batches = 0
            self._batches = iter(self.batches)
            batch = next(self._batches)

        return batch

    def render(self):
        # Assumes image dataset
        image = next(iter(self.sample().values())) if self.exp is None else self.exp.obs
        return np.array(image[random.randint(0, len(image) - 1)]).transpose(1, 2, 0)  # Channels-last

    # Arg-maxes if categorical distribution passed in  TODO Is this method needed besides rounding? Maybe move to env
    def adapt_to_discrete(self, action):
        shape = self.action_spec['shape']

        try:
            action = action.reshape(len(action), *shape)  # Assumes a batch dim
        except (ValueError, RuntimeError):
            try:
                action = action.reshape(len(action), -1, *shape)  # Assumes a batch dim
            except:
                raise RuntimeError(f'Discrete environment could not broadcast or adapt action of shape {action.shape} '
                                   f'to expected batch-action shape {(-1, *shape)}')
            action = action.argmax(1)

        discrete_bins, low, high = self.action_spec['discrete_bins'], self.action_spec['low'], self.action_spec['high']

        # Round to nearest decimal/int corresponding to discrete bins, high, and low  TODO Generalize to regression
        return np.round((action - low) / (high - low) * (discrete_bins - 1)) / (discrete_bins - 1) * (high - low) + low

    def __len__(self):
        return len(self.batches)


# Mean of empty reward should be NaN, catch acceptable usage warning  TODO delete
warnings.filterwarnings("ignore", message='.*invalid value encountered in scalar divide')
warnings.filterwarnings("ignore", message='invalid value encountered in double_scalars')
warnings.filterwarnings("ignore", message='Mean of empty slice')
