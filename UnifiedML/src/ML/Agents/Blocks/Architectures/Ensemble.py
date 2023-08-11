# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from copy import deepcopy

import torch
from torch import nn


class Ensemble(nn.Module):
    """
    "Ensembles" (stacks) multiple modules' outputs
    """
    def __init__(self, modules, dim=1):
        super().__init__()

        self.ensemble = nn.ModuleList([m if i == 0 or m != modules[i - 1]
                                       else deepcopy(m) for i, m in enumerate(modules)])
        self.dim = dim

        # This makes it possible to use model= syntax w ensembles
        if hasattr(modules[0], 'forward'):
            self.first = modules[0].forward

            if len(modules) > 1:
                # Not the most elegant. But works. Note: Using module[0] elsewhere might unexpectedly ensemble
                modules[0].forward = self.forward

    def forward(self, *x, **kwargs):
        return torch.stack([(getattr(self, 'first', m) if i == 0 else m)(*x, **kwargs)
                            for i, m in enumerate(self.ensemble)],
                           self.dim) if len(self) > 1 else self.ensemble[0](*x, **kwargs).unsqueeze(self.dim)

    def __len__(self):
        return len(self.ensemble)

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.ensemble[0], key)
