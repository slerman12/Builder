# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
minihydra / leviathan
A lightweight sys-arg manager, implemented from scratch by Sam Lerman.
See full hydra here: https://github.com/facebookresearch/hydra
"""

import ast
import importlib.util
import inspect
import itertools
import os.path
import re
import sys
import types
from collections import OrderedDict
from collections.abc import Mapping
from math import inf
import multiprocessing as mp
import yaml

app = '/'.join(str(inspect.stack()[-1][1]).split('/')[:-1])
cwd = os.getcwd()

sys.path.extend([app, cwd])  # Adding sys paths instead of module paths so that imports in modules work as well

yaml_search_paths = [app, cwd]  # List of paths to search for yamls in
module_paths = [app, cwd]  # List of paths to instantiate modules from
added_modules = {}  # Name: module pairs to instantiate from

task_dirs = ['', 'task/']  # Extra directories where tasks can be searched

log_dir = None


def get_module(_target_, paths=None, modules=None, recurse=False, try_again=False):
    if callable(_target_):
        # If target is a function or module already, just return target
        return _target_

    # Base search paths (formatted with '/' separated directories)
    paths = set(list(paths or []) + [path if '/' in path else path.replace('.', '/') for path in module_paths])

    # If target is a directory path
    if '/' in _target_:  # TODO Ideally, should support backslashes ('\') too
        # Example: arg=sub-directory/file.MyModule
        # Example: arg=sub-directory/file.py.MyModule

        # TODO agent=../../    (../../sub-directory/file.Agent)
        # TODO agent=../../ (../../sub-directory/file.py.Agent)
        # TODO Should accept full absolute path, not just relative to current path (directory/...)

        # Pull out everything before a backwards path, if anything
        *prefix, _target_ = _target_.rsplit('../', 1)
        # If the rest is a python file with extension specified and a module afterwards
        if '.py.' in _target_:
            # Get the path, module, and make sure extension still specified
            path, module_name = _target_.split('.py.', 1)
            path += '.py'
        else:
            # Otherwise, no module, just path, if extension specified and module isn't
            if '.py' in _target_:
                path, module_name = _target_, None
            else:
                # Or assume extension not specified, but last '.' specifies a module
                assert '.' in _target_, f'Directory path must include a .<module-name>, got: {_target_}'
                path, module_name = _target_.rsplit('.', 1)
                # Since "target is a directory path," we assume that the lack of an extension implies a init file
                path += '/__init__.py'

                # TODO Discriminate whether file/__init__.py or file.py
                # Example: arg=sub-directory/file.MyModule
                #   -> path = the rest of sub-directory/file/__init__.py  (this is wrong)
    else:
        # Example: arg=current_dir_file.MyModule
        # Example: arg=current_dir_file.py.MyModule

        # Since '/' not in target, assume target is a '.' path with the last '.' specifying a module
        prefix = None
        *path, module_name = _target_.rsplit('.py.', 1) if '.py.' in _target_ else _target_.rsplit('.', 1)
        # Convert the first parts into a file path, assuming the last of them to be a python extension
        # TODO Do this generally for path in all cases
        path = path[0].replace('..', '!@#$%^&*').replace('.', '/').replace('!@#$%^&*', '../') + '.py' if path else None
    # TODO e.g., path = as_directory(path)

    # Accept modules to search the subclasses of even when there are no paths to search from
    if modules is None:
        modules = Args()

    modules.update(added_modules)

    if path:
        keys = path.split('/')
        module = None

        first = keys[0].replace('.py', '')
        if keys and first in modules:
            # If first part of path is a module in modules, retrieve through that  TODO prefix? getattr error?
            module = modules[first]

            for key in keys[1:]:
                module = getattr(module, key.replace('.py', ''))
        else:
            # Import a module from an arbitrary directory s.t. it can be pickled! Can't use trivial SourceFileFolder
            for i, base in enumerate(paths.union({''})):
                base = os.path.abspath(base)
                if prefix:
                    base += '/' + prefix[0] + '..'  # Move relative backwards to base
                if base:
                    base += '/'

                if not os.path.exists(base + path):
                    if os.path.exists(base + path.replace('.py', '/__init__.py')):
                        path = path.replace('.py', '/__init__.py')
                    elif os.path.exists(base + path.replace('/__init__', '')):
                        path = path.replace('/__init__.py', '.py')
                    else:
                        continue

                path = path.replace('/', '.').replace('.py', '')

                # Check if cached
                if path in sys.modules:
                    module = sys.modules[path]
                else:
                    for key, value in sys.modules.items():
                        if hasattr(value, '__file__') and value.__file__ and base + path in value.__file__:
                            module = value
                            sys.modules[path] = module
                            break
                    else:
                        # Finally import
                        try:
                            module = importlib.import_module(path)
                        except ModuleNotFoundError:
                            *add, path = path.rsplit('.', 1)
                            sys.path.append(base + (add[0] if add else ''))
                            module = importlib.import_module(path)
                            path = (add[0] if add else '') + '.' + path
                        sys.modules[path] = module
                        break

        # If after this a module is not retrieved, can try again, sending in first part only,
        #   that somehow retrieving from an __init__.py file, and then subclassing into that
        if module is None:
            # Try one more possibility (_target_ refers to modules in an __init__.py file)
            if not try_again:
                # TODO Can make even more general by iterating through different depths of _target_ and module_names
                #     Currently supports the second-to-last being an __init__.py file
                _target_, *module_names = _target_.split('.')
                module = get_module(_target_, paths, modules, recurse, try_again=True)
                for name in module_names:
                    module = getattr(module, name)
                return module
            raise FileNotFoundError(f'Could not find path {path}. Search paths include: {paths}')
        else:
            # Return the relevant sub-module
            return module if module_name is None else getattr(module, module_name)
    elif module_name in modules:
        # Return the module from already-defined modules
        return modules[module_name]
    else:
        # See if module_name belongs to any of the modules
        #   e.g., if main_module in modules, arg=sub_module can reach main_module.sub_module
        for module in modules.values():
            if hasattr(module, module_name):
                return getattr(module, module_name)
        if not recurse:
            # See if module_name belongs to any of the paths
            #   e.g., via an __init__ file from that path
            e = None
            for path in paths:
                try:
                    return get_module(path + '.' + _target_, paths, modules, recurse=True)
                except Exception as e:
                    continue
            if e is not None:
                raise e
    raise FileNotFoundError(f'Could not find module {module_name}. Search modules include: {list(modules.keys())}')


class Args(Mapping):
    def __init__(self, _dict=None, **kwargs):
        self.__dict__.update({**(_dict or {}), **kwargs})

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return zip(self.keys(), self.values())

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return str(self.to_dict())

    def __len__(self):
        return len(self.__dict__)

    def get(self, key, *__default):
        return self.__dict__.get(key, *__default)

    def setdefault(self, key, *__default):
        return self.__dict__.setdefault(key, *__default)

    def pop(self, key, *__default):
        return self.__dict__.pop(key, *__default)

    def update(self, _dict=None, **kwargs):
        self.__dict__.update({**(_dict or {}), **kwargs})

    def to_dict(self):
        return {**{key: self[key].to_dict() if isinstance(self[key], Args) else self[key] for key in self}}


class Poo:
    def __init__(self):
        self.poo = '5'


"""
1. Convert paths to '/' and include current working directory at start. Each end with '/'
    $ path.replace(os.sep, '/') 
2. Go through target and pull out known_path and dot_paths. Convert former to '/'. Known path ends with '/' if no '.py'
3. If known path starts with '/', then treat paths as empty
4. Iterate through paths, and go through each dot path one by one
    - Except for last dot path, which should always be a module or sub-module, add as folder path and do:
        - Check 1: If python file, followed by module, followed by sub-modules if any further dots
            - This can be known if '.py' extension, and no iteration needed from the beginning
        - Check 2: If folder with __init__.py file followed by module, followed by sub-modules if any further dots
    - If any check throws an error, continue to trying the next check, proceeding with the iteration from 4
5. Iterate through named modules/added_modules 
    - Requires first dots or no dots of target be key in modules, then getattr
"""


def get_module_v2(_target_, paths=None, modules=None):
    if callable(_target_):
        # If target is a function or module already, just return target
        return _target_

    # 1. Convert paths and module paths to base search paths in '/' format and include current
    #    working directory at start. Each end with '/'

    # Paths and module paths formatted with '/' separated directories
    base_search_paths = set(list(paths or []) + [path.replace(os.sep, '/') if '/' in path.replace(os.sep, '/')
                                                 else path.replace('.', '/') for path in module_paths])

    # Make the paths absolute, include current working directory at start
    base_search_paths = set([os.path.abspath('')] + [os.path.abspath(base) for base in base_search_paths])

    # Each end with '/'
    base_search_paths = set([path + '/' if path[-1] != '/' else path for path in base_search_paths])

    # 2. Go through target and pull out known_path and dot_paths. Convert former to '/'.
    #    Known path ends with '/' if no '.py' extension

    _target_ = _target_.replace(os.sep, '/')

    # File path fully known if '.py' extension specified
    if '.py' in _target_:
        known_path, dot_path = _target_.rsplit('.py', 1)
        known_path += '.py'
    elif '/' in _target_:
        # Known path might be followed by the first part of dot path as the python file name,
        # or as a module in __init__.py in the known path directory
        known_path, dot_path = _target_.rsplit('/', 1)
        known_path += '/'  # Can be followed by __init__.py or first part of dot path + '.py'

        # __init__.py is known if dot path doesn't have any further dots that can be the module
        if '.' not in dot_path:
            known_path += '__init__.py'
    else:
        # Dot path here is ambiguous which parts are a file path and which parts are modules/sub-modules
        known_path, dot_path = None, _target_

    # 3. If known path starts with '/', then treat base search paths as empty

    if isinstance(known_path, str) and known_path[0] == '/':
        base_search_paths = {''}  # known path already includes the absolute path. No need to search base paths

    # 4. Iterate through base search paths, and go through each dot path one by one
    #     - Except for last dot path, which should always be a module or sub-module, add as folder path and do:
    #        - Check 1: If python file, followed by module, followed by sub-modules if any further dots
    #          - This can be known if '.py' extension, and no iteration needed from the beginning
    #        - Check 2: If folder with __init__.py file followed by module, followed by sub-modules if any further dots
    #     - If any check throws an error, continue to trying the next check, proceeding with the iteration from 4

    module = None

    # Iterate base paths
    for base in base_search_paths:
        dots = dot_path.split('.')

        for i, dot in enumerate(dots):  # TODO Might be able to nest deeper to avoid redundant dot iterations
            # If '.py' extension in target, file is known and can assume dots are all modules of that file
            if isinstance(known_path, str):
                if '.py' in known_path:
                    if module is None:
                        module = import_file(base + known_path)  # TODO Wrap in try-catch
                    module = getattr(module, dot)

                    # TODO break on last dot: i == len(dots) - 1
                    break
                else:
                    # Otherwise, first dot can be module in [known_path]__init__.py file or a python file itself.
                    # The rest of the dots must be modules/sub-modules
                    # TODO break on last dot: i == len(dots) - 1
                    break
            else:
                # Except for last dot, check each dot as possibly being a folder path,
                # and if modules/sub-modules follow
                # TODO break on last dot: i == len(dots) - 1
                break
    else:
        # 5. Iterate through named modules/added_modules if no path found

        # Accept modules to search the subclasses of even when there are no paths to search from
        if modules is None:
            modules = Args()

        modules.update(added_modules)

        dots = dot_path.split('.')

        # If first part of path is a module in modules, get module from that
        if dots[0] in modules:
            module = modules[dots[0]]

            for dot in dots[1:]:
                module = getattr(module, dot)
        else:
            # Otherwise, see if belongs as sub-module of any modules
            for module in modules.values():
                for dot in dots:
                    module = getattr(module, dot)  # TODO Try-catch-continue

    assert module is not None, ''  # TODO Error message

    # Return retrieved module
    return module


def rebuild(_target_, paths=None, modules=None, recurse=False, try_again=False):
    if callable(_target_):
        # If target is a function or module already, just return target
        return _target_

    # Base search paths (formatted with '/' separated directories)
    paths = set(list(paths or []) + [path if '/' in path else path.replace('.', '/') for path in module_paths])

    # Make the paths absolute, including no path
    paths = set([os.path.abspath(base) for base in paths.union('')] + [''])

    # If target is a directory path
    if '/' in _target_:  # TODO Ideally, should support backslashes ('\') too
        # Example: arg=sub-directory/file.MyModule
        # Example: arg=sub-directory/file.py.MyModule

        # TODO agent=../../    (../../sub-directory/file.Agent)
        # TODO agent=../../ (../../sub-directory/file.py.Agent)
        # TODO Should accept full absolute path, not just relative to current path (directory/...)

        # Allow './' paths
        if _target_[:2] == './':
            _target_ = _target_[2:]
            paths = {os.path.abspath('./')}

        # Pull out everything before a backwards path, if anything
        *backwards, remainder = _target_.rsplit('../', 1)

        # If the rest is a python file with extension specified and a module afterwards
        if '.py.' in _target_:
            # Get the path, module, and make sure extension still specified
            path, module_names = remainder.split('.py.', 1)
            if backwards:
                path = backwards[0] + '../' + path
            path += '.py'
        else:
            # Otherwise, figure out whether to add extension, or search __init__.py  TODO Maybe add to possibilities
            path, module_names = remainder.split('.', 1)
            if backwards:
                path = backwards[0] + '../' + path

            # Search all possible paths to see if python extension works
            for base in paths:
                separator = '' if base[-1] == '/' or path[0] == '/' else '/'

                if os.path.exists(base + separator + path + '.py'):
                    path += '.py'
                    break
            else:
                # Otherwise, assume __init__.py file
                separator = '' if path[-1] == '/' else '/'
                path += separator + '__init__.py'

                # Check if exists (Note: Code redundant to above, can make method)
                for base in paths:
                    separator = '' if base[-1] == '/' or path[0] == '/' else '/'

                    if os.path.exists(base + separator + path):
                        break
                else:
                    assert False, f'Could not find specified path {path.replace("/__init__.py", ".py")}, nor {path}.\n' \
                                  f'Searched: {paths}'

        # Import module from file, and get all sub-modules

        # In progress, could be all wrong

        for base in paths:
            separator = '' if base[-1] == '/' or path[0] == '/' else '/'

            # Find a path that exists
            if not os.path.exists(base + separator + path):
                continue

            # Check if module cached
            if base + separator + path + '.' + module_names in sys.modules:
                module = sys.modules[base + separator + path + '.' + module_names]
            else:
                # Check if module file cached
                for key, value in sys.modules.items():
                    if hasattr(value, '__file__') and value.__file__ and base + separator + path in value.__file__:
                        module = value
                        sys.modules[base + separator + path] = module
                        for key in module_names.split('.'):
                            module = getattr(module, key)
                        break
                else:
                    # Finally import
                    try:
                        module = importlib.import_module(base + separator + path)
                        for key in module_names.split('.'):
                            module = getattr(module, key)
                    except ModuleNotFoundError:
                        if base + separator + path not in sys.path:
                            sys.path.append(base + separator + path)
                        module, *keys = module_names.split()
                        for key in keys:
                            module = getattr(module, key)
                    sys.modules[base + separator + path + '.' + module_names] = module
                    break

            return module
    else:
        # Convert '.' to '/', until the last '.' that works as a '/' path, always treating the last '.' as a module
        #   If module not there, have to adaptively try __init__.py file

        # If there is no path, search modules
        return

"""
If I had to rebuild it from scratch, I would start by parsing. First, is when there's a '/' in the target. 
The question is, to assume that there is only one module, or to allow multiple modules. Well, in this case, can 
allow multiple modules since path and modules are disambiguated. So in this case, can separate out the path, get the
module from the corresponding file -- might have to be an __init__ file if no '.py' and last path part is only a 
directory -- and get the sub-modules via getattr from that. 

Another possibility is '.' in the target or neither '.' nor '/'. If '.', could be either a path or a module. 

How to disambiguate? Well, if the path exists, go with the path, except the last one -- the last one, always assume a 
module.

The paths can be either relative (to module_paths and specified paths) or absolute. First one that works.

Then, if the path doesn't exist, go through modules last. First, see if module exists. Then lastly, see if sub-modules 
exist. For example, at the very last step, PyTorch's Transformer will be instantiated. Otherwise, UnifiedML's because 
UnifiedML is specified as a search path or module. The paths take priority over the Pytorch nn module.
"""

if __name__ == '__main__':
    # 1. agent=current_dir_file.Agent
    # 2. agent=current_dir_file.py.Agent
    # 3. agent=sub-directory/[same things]
    # 4. agent=../prev_dir_file.Agent
    # 5. agent=../prev_dir_file.py.Agent
    # 6. agent=../../same-as-before.Agent and .py. Agent
    # 7. agent=/full-path with slashes only, no dots, except for the after the last file, with or without the .py
    # 8. Those files should allow __init__.py implicitly, or even not.
    # 9. There should be an inference priority order between file paths, added modules… added PyTorch modules?
    #    I’m iffy on this, but UnifiedML Transformer should be selected before nn.Transformer.
    # 10. And then there might be some hidden try-again inference situation.
    # 11. It should perhaps also support inference of lists without quotation marks

    print(rebuild('figuring_out.Poo'))


def get_module_BCE(_target_, paths=None, modules=None, recurse=False, try_again=False):
    if callable(_target_):
        # If target is a function or module already, just return target
        return _target_

    # Base search paths (formatted with '/' separated directories)
    paths = set(list(paths or []) + [path if '/' in path else path.replace('.', '/') for path in module_paths])

    # If target is a directory path
    if '/' in _target_:  # TODO Ideally, should support backslashes ('\') too
        # Example: arg=sub-directory/file.MyModule
        # Example: arg=sub-directory/file.py.MyModule

        # TODO agent=../../    (../../sub-directory/file.Agent)
        # TODO agent=../../ (../../sub-directory/file.py.Agent)
        # TODO Should accept full absolute path, not just relative to current path (directory/...)

        # Pull out everything before a backwards path, if anything
        *prefix, _target_ = _target_.rsplit('../', 1)
        # If the rest is a python file with extension specified and a module afterwards
        if '.py.' in _target_:
            # Get the path, module, and make sure extension still specified
            path, module_name = _target_.split('.py.', 1)
            path += '.py'
        else:
            # Otherwise, no module, just path, if extension specified and module isn't
            if '.py' in _target_:
                path, module_name = _target_, None
            else:
                # Or assume extension not specified, but last '.' specifies a module
                assert '.' in _target_, f'Directory path must include a .<module-name>, got: {_target_}'
                path, module_name = _target_.rsplit('.', 1)
                # Since "target is a directory path," we assume that the lack of an extension implies a init file
                path += '/__init__.py'

                # TODO Discriminate whether file/__init__.py or file.py
                # Example: arg=sub-directory/file.MyModule
                #   -> path = the rest of sub-directory/file/__init__.py  (this is wrong)
    else:
        # Example: arg=current_dir_file.MyModule
        # Example: arg=current_dir_file.py.MyModule

        # Since '/' not in target, assume target is a '.' path with the last '.' specifying a module
        prefix = None
        *path, module_name = _target_.rsplit('.py.', 1) if '.py.' in _target_ else _target_.rsplit('.', 1)
        # Convert the first parts into a file path, assuming the last of them to be a python extension
        # TODO Do this generally for path in all cases
        path = path[0].replace('..', '!@#$%^&*').replace('.', '/').replace('!@#$%^&*', '../') + '.py' if path else None
    # TODO e.g., path = as_directory(path)

    # Accept modules to search the subclasses of even when there are no paths to search from
    if modules is None:
        modules = Args()

    modules.update(added_modules)

    if path:
        keys = path.split('/')
        module = None

        first = keys[0].replace('.py', '')
        if keys and first in modules:
            # If first part of path is a module in modules, retrieve through that  TODO prefix? getattr error?
            module = modules[first]

            for key in keys[1:]:
                module = getattr(module, key.replace('.py', ''))
        else:
            # Import a module from an arbitrary directory s.t. it can be pickled! Can't use trivial SourceFileFolder
            for i, base in enumerate(paths.union({''})):
                base = os.path.abspath(base)
                if prefix:
                    base += '/' + prefix[0] + '..'  # Move relative backwards to base
                if base:
                    base += '/'

                if not os.path.exists(base + path):
                    if os.path.exists(base + path.replace('.py', '/__init__.py')):
                        path = path.replace('.py', '/__init__.py')
                    elif os.path.exists(base + path.replace('/__init__', '')):
                        path = path.replace('/__init__.py', '.py')
                    else:
                        continue

                path = path.replace('/', '.').replace('.py', '')

                # Check if cached
                if path in sys.modules:
                    module = sys.modules[path]
                else:
                    for key, value in sys.modules.items():
                        if hasattr(value, '__file__') and value.__file__ and base + path in value.__file__:
                            module = value
                            sys.modules[path] = module
                            break
                    else:
                        # Finally import
                        try:
                            module = importlib.import_module(path)
                        except ModuleNotFoundError:
                            *add, path = path.rsplit('.', 1)
                            sys.path.append(base + (add[0] if add else ''))
                            module = importlib.import_module(path)
                            path = (add[0] if add else '') + '.' + path
                        sys.modules[path] = module
                        break

        # If after this a module is not retrieved, can try again, sending in first part only,
        #   that somehow retrieving from an __init__.py file, and then subclassing into that
        if module is None:
            # Try one more possibility (_target_ refers to modules in an __init__.py file)
            if not try_again:
                # TODO Can make even more general by iterating through different depths of _target_ and module_names
                #     Currently supports the second-to-last being an __init__.py file
                _target_, *module_names = _target_.split('.')
                module = get_module(_target_, paths, modules, recurse, try_again=True)
                for name in module_names:
                    module = getattr(module, name)
                return module
            raise FileNotFoundError(f'Could not find path {path}. Search paths include: {paths}')
        else:
            # Return the relevant sub-module
            return module if module_name is None else getattr(module, module_name)
    elif module_name in modules:
        # Return the module from already-defined modules
        return modules[module_name]
    else:
        # See if module_name belongs to any of the modules
        #   e.g., if main_module in modules, arg=sub_module can reach main_module.sub_module
        for module in modules.values():
            if hasattr(module, module_name):
                return getattr(module, module_name)
        if not recurse:
            # See if module_name belongs to any of the paths
            #   e.g., via an __init__ file from that path
            e = None
            for path in paths:
                try:
                    return get_module(path + '.' + _target_, paths, modules, recurse=True)
                except Exception as e:
                    continue
            if e is not None:
                raise e
    raise FileNotFoundError(f'Could not find module {module_name}. Search modules include: {list(modules.keys())}')


# def get_module(_target_, paths=None, modules=None, recurse=False, try_again=False):
#     if callable(_target_):
#         # If target is a function or module already, just return target
#         return _target_
#
#     # Search paths
#     paths = set([next(iter(get_possible_paths(path).values())) for path in list(paths or []) + module_paths])
#
#     # Get possible target paths and module name(s) by searching search paths
#     paths_module_names = get_possible_paths(_target_, search_paths=paths)
#
#     # Search longest paths first
#     for path, module_names in sorted(paths_module_names.items(), key=len, reverse=True):
#         # '' designates None path
#         if path == '':
#             path = None
#
#         if path is None:
#             # Accept modules to search the subclasses of even when there are no paths to search from
#             if modules is None:
#                 modules = Args()
#
#             modules.update(added_modules)
#
#
# # Return possible file path and module names pairs
# def get_possible_paths(path, search_paths=None):
#     paths_module_names = Args()
#
#     # Go through the set of unique search paths
#     for i, base in enumerate(search_paths.union({''})):
#         # An absolute path can be defined from each
#         base = os.path.abspath(base)
#
#         # Split up path by single-dots
#
#         if not os.path.exists(base + path):
#             if os.path.exists(base + path.replace('.py', '/__init__.py')):
#                 path = path.replace('.py', '/__init__.py')
#             else:
#                 continue
#
#         path = path.replace('/', '.').replace('.py', '')
#
#
#
#     return paths_module_names
#


def instantiate(args, _i_=None, _paths_=None, _modules_=None, _signature_matching_=True, _override_=None, **kwargs):
    if hasattr(args, '_target_') or hasattr(args, '_default_') or \
            isinstance(args, dict) and ('_target_' in args or '_default_' in args):
        args = Args(args)

        if '_overload_' in args:
            kwargs.update(args.pop('_overload_'))  # For overriding args without modifying defaults

        while '_default_' in args:  # Allow inheritance between sub-args
            args = Args(_target_=args['_default_']) if isinstance(args['_default_'], str) \
                else {**args.pop('_default_'), **args}

        if '_if_not_null_' in args:  # Allows conditional overriding if values aren't None
            args.update({key: value for key, value in args.pop('_if_not_null_').items() if value is not None})

        _target_ = args.pop('_target_')

        _overrides_ = args.pop('_overrides_') if '_overrides_' in args else {}

        if _target_ is None:
            return
        elif isinstance(_target_, str) and '(' in _target_ and ')' in _target_:  # Function calls
            for key in kwargs:
                _target_ = _target_.replace(f'kwargs.{key}', f'kwargs["{key}"]')  # Interpolation
            module = eval(_target_, None, {**added_modules, **(_modules_ or {}), 'kwargs': kwargs})  # Direct code exec
        else:
            module = _target_

            if isinstance(module, str):
                module = get_module(module, _paths_, _modules_)

            if isinstance(module, type) or inspect.isfunction(module):
                # Signature matching
                args.update(kwargs)
                signature = inspect.signature(module).parameters if _signature_matching_ else args.keys()
                args = args if 'kwargs' in signature else {key: args[key] for key in args.keys() & signature}
                module = module(**args)

        for key, value in _overrides_.items():  # Override class functions
            if value is not None:
                setattr(module, key, types.MethodType(get_module(value) if isinstance(value, str) else value, module))
    else:
        # Convert to config
        return instantiate(Args(_target_=args), _i_, _paths_, _modules_, _signature_matching_, _override_, **kwargs)

    # Allow sub-indexing (if specified)
    return module[_i_] if (isinstance(module, (list, tuple)) or 'ModuleList' in str(type(module))) and _i_ is not None \
        else module


# Check if a string path to a module is valid for instantiation
def valid_path(path, dir_path=False, module_path=True, module=True, _modules_=None):
    truth = False

    if not isinstance(path, str):
        return truth

    if dir_path:
        try:
            truth = os.path.exists(path)
        except FileNotFoundError:
            pass

    if module_path and not truth and path.count('.') > 0:
        # *root, file, _ = path.replace('.', '/').rsplit('/', 2)
        *root, file, _ = path.replace('.', '.' if '/' in path else '/').rsplit('/', 2)  # TODO Recently changed to this
        root = root[0].strip('/') + '/' if root else ''
        for base in module_paths:
            if '/' not in base:
                base = base.replace('.', '/')  # TODO Recently added this
            try:
                truth = os.path.exists(base + '/' + root + file + '.py')
            except FileNotFoundError:
                break
            if truth:
                break

    if _modules_ is None:
        _modules_ = {}

    _modules_.update(added_modules)

    if module and not truth:
        sub_module, *sub_modules = path.split('.')

        if sub_module in _modules_:
            sub_module = _modules_[sub_module]

            try:
                for key in sub_modules:
                    sub_module = getattr(sub_module, key)
                truth = True
            except AttributeError:
                pass

    return truth


def open_yaml(source, return_path=False):
    for path in yaml_search_paths + ['']:
        try:
            with open(path + '/' + source.strip('/'), 'r') as file:
                args = yaml.safe_load(file)
            return (recursive_Args(args), path + '/' + source.strip('/')) if return_path else recursive_Args(args)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f'{source} not found. Searched: {yaml_search_paths + [""]}')


# Allow access via attributes recursively
def recursive_Args(args):
    if isinstance(args, (Args, dict)):
        args = Args(args)

    items = enumerate(args) if isinstance(args, list) \
        else args.items() if isinstance(args, Args) else ()  # Iterate through lists, tuples, or dicts

    for key, value in items:
        args[key] = _parse(recursive_Args(value))  # Recurse through inner values

    return args


def recursive_update(original, update, _target_inference=True):
    for key, value in update.items():
        if isinstance(value, (Args, dict)) and key in original and isinstance(original[key], (Args, dict)) and value:
            original[key].update(recursive_update(original[key], value))
            # TODO Ideally, {...} would override, while .x= or -->x: would update.
            #      How to parse {...} specially? For now, just checking non-empty via "and value".
        elif key in original and isinstance(original[key], (Args, dict)) and '_target_' in original[key] \
                and not (isinstance(value, (dict, Args)) and '_target_' in value) and _target_inference:
            original[key]['_target_'] = value  # Infer value as _target_
        else:
            original[key] = value
    return original


# Search combinations of specified task dirs
def add_task_dirs(*paths):
    global task_dirs
    search = ('task',) + paths
    task_dirs = ['']

    for combo in range(1, len(search) + 1):
        for subset in itertools.combinations(search, combo):
            task_dirs.append('/'.join(subset) + '/')

    for combo in range(2, len(search) + 1):
        for subset in itertools.combinations(reversed(search), combo):
            task_dirs.append('/'.join(subset) + '/')


def get_task(path):
    task = None
    for i, task_dir in enumerate(task_dirs):
        try:
            task_path = task_dir + path.replace('.yaml', '') + '.yaml'
            task = read(task_path, recurse=True, save_task_path=True)
            break
        except FileNotFoundError:
            try:
                task = path.split('/')
                task_path = '/'.join(task[:-1]) + '/' + task_dir + task[-1].replace('.yaml', '') + '.yaml'
                task = read(task_path, recurse=True, save_task_path=True)
                break
            except FileNotFoundError:
                if i == len(task_dirs) - 1:
                    raise FileNotFoundError(f'Could not find task {path}. '
                                            f'Searched: {task_dirs} in {yaml_search_paths}')
    return task


def read(source, recurse=False, save_task_path=False):
    args, path = open_yaml(source, return_path=True)

    # Parse pseudonyms
    if '_pseudonyms_' in args:
        # Pseudonyms are alternate interface-keys. Note they are not interchangeable in the args struct,
        # only as an interface. The primary should be used in the in-code retrieved args structs.
        for primary, keys in args._pseudonyms_.items():
            value = get(args, primary, resolve=False)
            setdefault(args, args._pseudonyms_[primary][0], value)
            preceding = args._pseudonyms_[primary][0]
            for key in keys[1:] + [primary]:
                if isinstance(value, (Args, dict)):
                    setdefault(args, key, Args(_default_=f'${{{preceding}}}'))
                else:
                    setdefault(args, key, f'${{{preceding}}}')
                preceding = key

    path = os.path.dirname(os.path.abspath(path))

    # Add task project-directory to system paths
    if save_task_path:
        add = path

        # Step out of task dirs
        while add.split('/')[-1] + '/' in task_dirs:
            add = os.path.dirname(add)
        if add not in sys.path:
            sys.path.append(add)
        if add not in yaml_search_paths:
            yaml_search_paths.append(add)
        if add not in module_paths:
            module_paths.append(add)

    # Need to allow imports
    if 'imports' in args:
        imports = args.pop('imports')

        if 'self' not in imports:
            imports.append('self')

        self = recursive_Args(args)

        added = None
        for module in imports:
            if path not in sys.path:
                added = path
                yaml_search_paths.append(path)
            module = self if module == 'self' else get_task(module)
            if added:
                yaml_search_paths.pop(yaml_search_paths.index(added))
                added = None
            recursive_update(args, module)

    # Parse task
    if not recurse:
        if 'task' in portal:
            args['task'] = portal['task']

        for sys_arg in sys.argv[1:]:
            key, value = sys_arg.split('=', 1)
            if key == 'task':
                args['task'] = value

    # Command-line task import
    if 'task' in args and args.task not in [None, 'null']:
        if '/' not in args.task:
            args.task = args.task.replace('.yaml', '').replace('.', '/')

        # Locate tasks, adding relative cross-task paths temporarily
        added = None

        if path not in sys.path:
            added = path
            yaml_search_paths.append(path)
        task = get_task(args.task)
        if added:
            yaml_search_paths.pop(yaml_search_paths.index(added))
        recursive_update(args, task)

    return args


def _parse(value):
    if isinstance(value, str):
        if re.compile(r'^\[.*\]$').match(value) or \
                re.compile(r'^\(.*\)$').match(value) or \
                re.compile(r'^\{.*\}$').match(value) or \
                re.compile(r'^-?[0-9]*.?[0-9]+(e-?[0-9]*.?[0-9]+)?$').match(value):
            value = ast.literal_eval(value)  # TODO Custom with no quotes required for strings
        elif isinstance(value, str) and value.lower() in ['true', 'false', 'null', 'inf']:
            value = True if value.lower() == 'true' else False if value.lower() == 'false' \
                else None if value.lower() == 'null' else inf
    return value


def parse(args=None):
    # Parse portal
    global portal
    all_args = OrderedDict(**portal)  # TODO Task args should be popped from portal / task should have priority

    # Parse command-line
    for sys_arg in sys.argv[1:]:
        keys, value = sys_arg.split('=', 1)
        all_args[keys] = value

    # Parse portal and command-line  Note: portal syntax parallels command-line
    for keys, value in all_args.items():
        keys = keys.split('.')
        value = _parse(value)
        # Iterate through Arg depths up to second-to-last key
        base_value = args
        for i, key in enumerate(keys[:-1]):
            # Iterate through keys whose depths don't yet exist (but should)
            if key not in base_value:
                join_keys = '.'.join(keys)
                sets_this_depth = join_keys in all_args.keys()
                sets_next_depth = join_keys + '.' + keys[i + 1] in all_args.keys()
                # If system args immediately sets a next depth
                if sets_this_depth and sets_next_depth:
                    # Treat this non-existent depth as an instantiation Arg
                    base_value[key] = Args(_target_=None)
                else:
                    # Otherwise, just make sure depth exists
                    base_value[key] = Args()
            # If the depth exists but not as Arg
            elif not isinstance(base_value[key], (Args, dict)):
                base_value[key] = Args(_target_=base_value[key])  # Treat as _target_
            # Next depth
            base_value = base_value[key]  # An Arg
        # Special handling of instantiable args depending on presence of _target_
        key = keys[-1]
        # If instantiation Arg pre-exists in Args for this key
        if key in base_value and isinstance(base_value[key], (Args, dict)) and '_target_' in base_value[key]:
            # If value itself is an Arg or dict
            if isinstance(value, (Args, dict)):
                if '_target_' in value:
                    base_value[key] = value  # Override if _target_ in value
                else:
                    base_value[key].update(value)  # Update if _target_ not in value
            else:
                base_value[key]['_target_'] = value  # Set value as _target_
        else:
            # Otherwise, just set value to Arg normally, not worrying about instantiation Arg _target_
            base_value[key] = value

    return args


def get(args, keys, resolve=True):
    arg = args
    keys = keys.split('.')
    for key in keys:
        arg = getattr(arg, key)
    return interpolate([arg], args)[0] if resolve else arg  # Interpolate to make sure gotten value is resolved


# Set a default in a dict but multi-depth from a string of dot-separated keys, or return existing value
def setdefault(args, keys, default):
    arg = args
    keys = keys.split('.')
    for key in keys[:-1]:
        arg = arg.setdefault(key, Args())
    if keys[-1] not in arg:
        arg[keys[-1]] = default
    return arg[keys[-1]]


# minihydra.grammar.append(rule)
grammar = []  # List of funcs


def interpolate(arg, args=None, **kwargs):
    if isinstance(arg, Args):
        recursive_update(arg, kwargs)  # Note: Doesn't create new dicts if pre-existing
    # Perhaps instead do:
    # for key, value in kwargs.items():
    #     arg[key] = value

    if args is None:
        args = arg

    def _interpolate(match_obj):
        if match_obj.group() is not None:
            try:
                got = get(args, match_obj.group()[2:][:-1])
                # Try this for classes and types
                out = got.__name__ if inspect.isclass(got) \
                    else type(got).__name__ if not isinstance(got, (str, int, float, tuple, list, set, bool)) \
                    else str(got)
                # out = str(got)
                if out == '???':
                    return str(match_obj.group())
                return out
            except AttributeError:
                pass

    items = enumerate(arg) if isinstance(arg, (list, tuple)) \
        else arg.items() if isinstance(arg, Args) else ()  # Iterate through lists, tuples, or dicts

    for key, value in items:
        if isinstance(value, str):
            if re.compile(r'.+\$\{[^((\$\{)|\})]+\}.*').match(value) or \
                    re.compile(r'.*\$\{[^((\$\{)|\})]+\}.+').match(value):
                arg[key] = re.sub(r'\$\{[^((\$\{)|\})]+\}', _interpolate, value)  # Strings
            elif re.compile(r'\$\{[^((\$\{)|\})]+\}').match(value):
                try:
                    out = get(args, value[2:][:-1])
                    if not (isinstance(out, str) and out == '???'):
                        arg[key] = out  # Objects
                except AttributeError:
                    pass

        for rule in grammar:
            if isinstance(value, str):
                arg[key] = rule(arg[key])

        if isinstance(arg[key], (list, tuple, Args)):
            interpolate(arg[key], args)  # Recurse through inner values

    return arg


# yaml-able representation of args
def to_repr(args):
    for key, value in args.items():
        if not isinstance(value, (list, tuple, str, float, int)) and value is not None:
            args[key] = to_repr(value) if isinstance(value, (dict, Args)) else str(value)

    return args


def log(args):
    if 'minihydra' in args:
        if 'log_dir' in args.minihydra:
            path = interpolate([args.minihydra.log_dir], args)[0] + '.yaml'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as file:
                args = interpolate(parse(Args()), args)
                args.update(_minihydra_={'app': app, 'cwd': cwd})
                yaml.dump(to_repr(args).to_dict(), file, sort_keys=False)


def multirun(args):
    # Divide args into multiple copies
    pass


# Can just get args, no decorator
def just_args(source=None, logging=False):
    args = Args() if source is None else read(source)
    args = parse(args)
    args = interpolate(args)  # Command-line requires quotes for interpolation
    if logging:
        log(args)

    return args


portal = {}


# Those args remain in cache for all future args retrievals unless manually reset
def reset_portal():
    set_portal()


# A code-based interface for setting args
def set_portal(args=None, **keywords):
    global portal
    portal = {**(args or {}), **keywords}


def len_return_variables(func):
    return len(re.findall(r'return\s*(.*)\n*$', inspect.getsourcelines(func)[0][-1])[0].split(',')) or 1


# Can decorate a method with args in signature
def get_args(source=None, logging=True):
    def decorator_func(func):
        def main(_args_=None, __main__=False, **_kwargs_):
            # If __main__, only run in __main__ call and MainProcess, not imports or forks
            __main__ = not __main__ or sys._getframe(1).f_globals["__name__"] == '__main__' \
                       and mp.current_process().name == 'MainProcess'

            if __main__:
                set_portal(_args_, **_kwargs_)
                return func(just_args(source, logging=logging))

            # If not running, still try to return the correct number of outputs; assumes consistent return sizes
            return (None,) * len_return_variables(func)

        return main

    return decorator_func
