# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
This file makes it possible to import UnifiedML as a package or launch it within Python.
"""

import os
import sys
import inspect

# import torch

UnifiedML = os.path.dirname(__file__)
app = '/'.join(str(inspect.stack()[-1][1]).split('/')[:-1])


# Imports UnifiedML paths and the paths of any launching app
def import_paths():
    sys.path.append(UnifiedML)

    from Hyperparams.minihydra import yaml_search_paths

    if UnifiedML not in yaml_search_paths:
        yaml_search_paths.append(UnifiedML)  # Adds UnifiedML to search path

    # Adds Hyperparams dir to search path
    for path in [UnifiedML, app, os.getcwd()]:
        if path + '/Hyperparams' not in yaml_search_paths and os.path.exists(path + '/Hyperparams'):
            yaml_search_paths.append(path + '/Hyperparams')


import_paths()


# Launches UnifiedML from inside a launching app with specified args  TODO Move to Utils since MT needs access
def launch(**args):
    import Utils
    from Run import main

    original = list(sys.argv)

    command_line_args = {arg.split('=')[0] for arg in sys.argv if '=' in arg}
    added = set()

    for key, value in args.items():
        if isinstance(value, (str, bool)):
            if key not in command_line_args:
                sys.argv.insert(-2, f'{key}={value}')  # For minihydra grammars in Utils  TODO Maybe just interpolate
                added.add(key)

    Utils.launch_args = {key: args[key] for key in args.keys() - command_line_args - added}

    # torch.multiprocessing.set_start_method('spawn')

    main()  # Run

    Utils.launch_args = {}
    sys.argv = original
