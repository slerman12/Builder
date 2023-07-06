# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
UnifiedML - A unified framework for intelligence training. Built and conceived by Sam Lerman.

    This file makes it possible to import UnifiedML as a package or launch it within Python.
"""

import sys
import os

from torch import multiprocessing as mp


sys.path.append(os.path.dirname(__file__))

_, dirs, files = next(os.walk(os.path.dirname(__file__)))

for file in files:
    if file != '__init__.py' and file[-3:] == '.py':
        globals().update({file[:-3]: __import__(file[:-3])})

for dir in dirs:
    globals().update({dir: __import__(dir)})

from Utils import launch
main = launch  # TODO main
