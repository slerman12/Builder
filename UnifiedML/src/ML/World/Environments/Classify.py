# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import random
import os
import warnings

import numpy as np
import torch

from torch.utils.data import DataLoader

from World.Dataset import load_dataset, worker_init_fn, compute_stats
from minihydra import Args


class Classify:
    """
    A general-purpose environment:

    Must accept: **kwargs as init arg.

    Must have:

    (1) a "step" function, action -> exp
    (2) "reset" function, -> exp
    (3) "render" function, -> image
    (4) "episode_done" attribute
    (5) "obs_spec" attribute which includes:
        - "shape", "mean", "stddev", "low", "high" (the last 4 can be None)
    (6) "action-spec" attribute which includes:
        - "shape", "discrete_bins" (should be None if not discrete), "low", "high", and "discrete"
    (7) "exp" attribute containing the latest exp

    Recommended: Discrete environments should have a conversion strategy for adapting continuous actions (e.g. argmax)

    An "exp" (experience) is an Args consisting of "obs", "action" (prior to adapting), "reward", and "label"
    as numpy arrays with batch dim or None. "reward" is an exception: should be numpy array, can be empty/scalar/batch.

    ---

    Extended to accept a "Dataset=" config arg, which instantiates a Dataset. Datasets must:
    - extend Pytorch Datasets
    - output (obs, label) pairs

    Datasets can:
    - include a "classes" attribute that lists the different class names or classes

    The "step" function has a no-op default action (action=None) to allow for Offline-mode streaming.

    An "evaluate_episodes" attribute divides evaluation across batches since batch=episode in this environment.

    """
    def __init__(self, dataset, test_dataset=None, train=True, offline=True, generate=False,
                 batch_size=8, num_workers=1, low=None, high=None, device='cpu', **kwargs):
        self.episode_done = False

        if not train:
            # Inherit from test_dataset
            if test_dataset._target_ is not None:
                dataset = test_dataset
            elif test_dataset.subset is not None:
                dataset.subset = test_dataset.subset
            elif test_dataset.transform._target_ is not None:
                dataset.transform = test_dataset.transform

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

        self._batches = iter(self.batches)

        # Check shape of x
        obs_shape = tuple(dataset[0][0].shape)
        obs_shape = (1,) * (2 - len(obs_shape)) + obs_shape  # At least 1 channel dim and spatial dim - can comment out

        self.action_spec = Args({'shape': (1,),
                                 'discrete_bins': len(dataset.classes),
                                 'low': 0,
                                 'high': len(dataset.classes) - 1,
                                 'discrete': True})

        self.obs_spec = Args({'shape': obs_shape,
                              'low': low,
                              'high': high})

        # Fill in necessary obs_spec and action_spc stats from dataset  TODO Only when norm or standardize
        if train and (offline or generate):
            self.obs_spec.update(compute_stats(self.batches))

        # TODO Alt, load_dataset can output Args of recollected stats as well; maybe specify what to save in card replay

        self.exp = None

        self.evaluate_episodes = len(self.batches)

    def step(self, action=None):
        # No action - "no-op" - for Offline streaming
        if action is None:
            self.reset()  # Sample new batch
            return self.exp  # Return new batch

        # Adapt to discrete!
        _action = self.adapt_to_discrete(action)

        correct = (self.exp.label == _action).astype('float32')

        self.exp.reward = correct
        self.exp.action = action  # Note: can store argmax instead

        self.exp.done = self.episode_done = True

        return self.exp

    def reset(self):  # The reset step is never stored in isolation
        obs, label = [np.array(b, dtype='float32') for b in self.sample()]
        if len(label.shape) == 1:
            label = np.expand_dims(label, 1)

        batch_size = obs.shape[0]

        obs.shape = (batch_size, *self.obs_spec['shape'])

        self.episode_done = False

        # Create experience
        exp = {'obs': obs, 'label': label, 'done': False}

        self.exp = Args(exp)  # Experience

        # if self.generate:
        #     exp.obs = exp.label = np.array([batch_size, 0])  # Can disable for generative

        return self.exp

    def render(self):
        # Assumes image dataset
        image = self.sample()[0] if self.exp is None else self.exp.obs
        return np.array(image[random.randint(0, len(image) - 1)]).transpose(1, 2, 0)  # Channels-last

    def sample(self):
        try:
            return next(self._batches)
        except StopIteration:
            self._batches = iter(self.batches)
            return next(self._batches)

    # Arg-maxes if categorical distribution passed in
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

        # Round to nearest decimal/int corresponding to discrete bins, high, and low
        return np.round((action - low) / (high - low) * (discrete_bins - 1)) / (discrete_bins - 1) * (high - low) + low


# Mean of empty reward should be NaN, catch acceptable usage warning
warnings.filterwarnings("ignore", message='.*invalid value encountered in scalar divide')
warnings.filterwarnings("ignore", message='invalid value encountered in double_scalars')
warnings.filterwarnings("ignore", message='Mean of empty slice')
