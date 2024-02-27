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
    - have a step(action) function
    - have a reset() function

        The step() and reset() functions output experiences ("exp")
            i.e. dicts containing datums like "obs", "action", "reward", 'label", "done"

        The end of an episode or epoch may be marked by the "done" boolean. If unspecified, it will be assumed True
        during step() calls and False during reset() calls.
        This can be useful for tabulating metrics based on multiple steps of batches
        or organizing temporal data in Replay.
        - The "done" step doesn't get acted on. Its contents can be a no-op (either output None or just {'done': True}),
        or can include any additional datums needed for logging or storing in Replay, such as "reward".
        - The reset() state can't ever be a "done" state even if you specify it as such. In that case,
        output None from your step() function.

        Instead of outputting one experience, can output a (prev, now) pair of experiences. Some datums, such as reward,
        are time-delayed and specifying them in a separate "prev" experience makes it possible to pair corresponding
        time-step pairs for computing metrics easier.

    Envs can:
    - have an obs_spec dict
    - have an action_spec dict

        depending on what can or can't be inferred. For example, obs_spec.shape often isn't necessary
        since it can be inferred from the "obs" output of the reset() function. action_spec.shape from a "label" if
        present.

        For obs_spec, these stats can include: "shape", "mean", "stddev", "low", "high".
        Useful for standardization and normalization.

        For action_spec, these stats can include: "shape", "discrete_bins", "low", "high", "discrete".
        For discrete action spaces and action normalization. Many of these can be inferred,
        and see Environment.py "action_spec" @property for what the defaults are if left unspecified.

    ---

    Datasets must:
    - extend Pytorch Datasets
    - output (obs, label) pairs, or dicts of named datums, e.g., {'obs': obs, 'label': label, ...}

    Datasets can:
    - include a "classes" attribute that lists the different class names or classes
        This allows quick and exact computation of number of classes being predicted from, without having to count
        via iterating through the whole dataset.
    """
    def __init__(self, dataset, test_dataset=None, train=True, offline=True, generate=False, batch_size=8,
                 num_workers=1, standardize=False, norm=False, device='cpu', **kwargs):
        if not train and test_dataset is not None:
            # Assume test_dataset
            dataset = test_dataset

        # (This might not generally be necessary)
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)  # Shared memory can create a lot of file descriptors
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))  # Increase soft limit to hard limit

        dataset = load_dataset('World/ReplayBuffer/Offline/', dataset, allow_memory=False, train=train)

        discrete = hasattr(dataset, 'classes')  # load_dataset adds a .classes attribute to the dataset if discrete

        # CPU workers
        num_workers = max(1, min(num_workers, os.cpu_count()))

        self.batches = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=torch.device(device).type == 'cuda',
                                  collate_fn=getattr(dataset, 'collate_fn', None),  # Useful if streaming dynamic lens
                                  worker_init_fn=worker_init_fn)  # TODO Support using Replay in Env (and by default)

        self._batches = iter(self.batches)

        self.num_sampled_batches = 0

        # Experience
        self.exp = None

        # Fill in necessary obs_spec and action_spec stats (e.g. mean, stddev, low, high) from dataset
        if train and (offline or generate) and (standardize or norm):
            # TODO Alternatively, load_dataset can output Args of recollected stats as well;
            #  maybe specify what to save in card replay
            self.obs_spec = compute_stats(self.batches)  # TODO Should provide per-Datum specs

        if discrete:
            self.action_spec = {'discrete_bins': len(dataset.classes)}

    def step(self, action):
        if self.num_sampled_batches < len(self):  # Episode is "done" at end of epoch
            return self.reset()  # Re-sample datums

    def reset(self):
        # Sample batch
        self.exp = Args(done=False, **self.sample())
        self.num_sampled_batches += 1

        # Convert sample to numpy  TODO Maybe not necessary
        for key, value in self.exp.items():
            self.exp[key] = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.array(value)

        # Label should have shape (batch_size, 1)
        if 'label' in self.exp and len(self.exp.label.shape) == 1:
            self.exp.label = np.expand_dims(self.exp.label, 1)

        # Return batch
        return self.exp

    def sample(self):
        try:
            return next(self._batches)
        except StopIteration:
            self.num_sampled_batches = 0
            self._batches = iter(self.batches)
            return next(self._batches)

    def render(self):
        # Assumes image dataset
        image = next(iter(self.sample().values())) if self.exp is None else self.exp.obs
        return np.array(image[random.randint(0, len(image) - 1)]).transpose(1, 2, 0)  # Channels-last

    def __len__(self):
        return len(self.batches)


# Mean of empty reward should be NaN, catch acceptable usage warning  TODO Delete if these warnings don't pop up anyway
warnings.filterwarnings("ignore", message='.*invalid value encountered in scalar divide')
warnings.filterwarnings("ignore", message='invalid value encountered in double_scalars')
warnings.filterwarnings("ignore", message='Mean of empty slice')
