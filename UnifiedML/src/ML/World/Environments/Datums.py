# Copyright (c) AGI.__init__. All Rights Reserved.
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
    A general-purpose environment must have a step function, reset function, obs_spec, and action_spec

    The step and reset functions output "experiences"
        i.e. dicts containing datums like "obs", "reward", 'label", "done"

    ---

    Datasets must:
    - extend Pytorch Datasets
    - output (obs, label) pairs

    Datasets can:
    - include a "classes" attribute that lists the different class names or classes

    The "step" function has a no-op default action (action=None) to allow for Offline-mode streaming.
    """
    def __init__(self, dataset, test_dataset=None, train=True, offline=True, generate=False, batch_size=8,
                 num_workers=1, low=None, high=None, standardize=False, norm=False, device='cpu', **kwargs):
        if not train:
            # Inherit from test_dataset
            if test_dataset._target_ is not None:
                dataset = test_dataset
            elif test_dataset.subset is not None:
                dataset.subset = test_dataset.subset
            elif test_dataset.transform._target_ is not None:
                dataset.transform = test_dataset.transform

        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)  # Shared memory can create a lot of file descriptors
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))  # Increase soft limit to hard limit

        dataset = load_dataset('World/ReplayBuffer/Offline/', dataset, allow_memory=False, train=train)

        # CPU workers
        num_workers = max(1, min(num_workers, os.cpu_count()))

        self.batches = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=torch.device(device).type == 'cuda',
                                  collate_fn=getattr(dataset, 'collate_fn', None),  # Useful if streaming dynamic lens
                                  worker_init_fn=worker_init_fn)

        self.num_sampled_batches = 0

        self._batches = iter(self.batches)

        # Check shape of x
        obs_shape = tuple(dataset[0][0].shape)
        obs_shape = (1,) * (2 - len(obs_shape)) + obs_shape  # At least 1 channel dim and spatial dim - can comment out

        self.discrete = hasattr(dataset, 'classes')

        action_shape = (1,) if self.discrete or not hasattr(dataset[0][1], 'shape') else tuple(dataset[0][1].shape)

        self.action_spec = Args({'shape': action_shape,
                                 'discrete_bins': len(dataset.classes) if self.discrete else None,
                                 'low': 0 if self.discrete else None,
                                 'high': len(dataset.classes) - 1 if self.discrete else None,
                                 'discrete': self.discrete})

        self.obs_spec = Args({'shape': obs_shape,
                              'low': low,
                              'high': high})

        # Fill in necessary obs_spec and action_spc stats from dataset  TODO Only when norm or standardize
        if train and (offline or generate) and (standardize or norm):
            self.obs_spec.update(compute_stats(self.batches))

        # TODO Alt, load_dataset can output Args of recollected stats as well; maybe specify what to save in card replay

        # Experience
        self.exp = None

    def step(self, action=None):
        if self.num_sampled_batches > 1:  # To account for reset()
            self.reset()  # Re-sample

        self.num_sampled_batches += 1

        if action is not None:  # No action - "no-op" - allowed for Offline streaming
            # Adapt to discrete!    Note: storing argmax
            self.exp.action = self.adapt_to_discrete(action) if self.discrete else action

        return self.exp

    def reset(self):  # The reset step is never stored
        obs, label = [np.array(b, dtype='float32') for b in self.sample()]  # Sample

        if len(label.shape) == 1:
            label = np.expand_dims(label, 1)

        batch_size = obs.shape[0]

        obs.shape = (batch_size, *self.obs_spec['shape'])

        # Create experience
        self.exp = Args(obs=obs, label=label, done=self.num_sampled_batches == len(self))

        return self.exp

    def sample(self):
        try:
            return next(self._batches)
        except StopIteration:
            self.num_sampled_batches = 1
            self._batches = iter(self.batches)
            return next(self._batches)

    def render(self):
        # Assumes image dataset
        image = self.sample()[0] if self.exp is None else self.exp.obs
        return np.array(image[random.randint(0, len(image) - 1)]).transpose(1, 2, 0)  # Channels-last

    # Arg-maxes if categorical distribution passed in  TODO Is this method needed besides rounding? Move to env
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
