# Copyright (c) AGI.__init__. All Rights Reserved.
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

task_dirs = ['', 'task']  # Extra directories where tasks can be searched

log_dir = None


def get_module(_target_, paths=None, modules=None):
    if callable(_target_):
        return _target_

    paths = list(paths or []) + module_paths

    if modules is None:
        modules = Args()

    modules.update(added_modules)

    if '/' in _target_:
        *prefix, _target_ = _target_.rsplit('../', 1)
        if '.py.' in _target_:
            path, module_name = _target_.split('.py.', 1)
            path += '.py'
        else:
            if '.py' in _target_:
                path, module_name = _target_, None
            else:
                assert '.' in _target_, f'Directory path must include a .<module-name>, got: {_target_}'
                path, module_name = _target_.rsplit('.', 1)
                path += '/__init__.py'
    else:
        prefix = None
        *path, module_name = _target_.rsplit('.', 1)
        path = path[0].replace('..', '!@#$%^&*').replace('.', '/').replace('!@#$%^&*', '../') + '.py' if path else None

    if path:
        keys = path.split('/')
        module = None

        first = keys[0].replace('.py', '')
        if keys and first in modules:
            module = modules[first]

            for key in keys[1:]:
                module = getattr(module, key.replace('.py', ''))
        else:
            # Import a module from an arbitrary directory s.t. it can be pickled! Can't use trivial SourceFileFolder
            for i, base in enumerate(paths + ['']):
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
        if module is None:
            raise FileNotFoundError(f'Could not find path {path}. Search paths include: {paths}')
        else:
            # Return the relevant sub-module
            return module if module_name is None else getattr(module, module_name)
    elif module_name in modules:
        # Return the module from already-defined modules
        return modules[module_name]
    else:
        for module in modules.values():
            if hasattr(module, module_name):
                return getattr(module, module_name)
    raise FileNotFoundError(f'Could not find module {module_name}. Search modules include: {list(modules.keys())}')


def instantiate(args, _i_=None, _paths_=None, _modules_=None, _signature_matching_=True, _override_=None, **kwargs):
    if hasattr(args, '_target_') or hasattr(args, '_default_') or \
            isinstance(args, dict) and ('_target_' in args or '_default_' in args):
        args = Args(args)

        if '_overload_' in args:
            kwargs.update(args.pop('_overload_'))  # For overriding args without modifying defaults

        while '_default_' in args:  # Allow inheritance between sub-args
            args = Args(_target_=args['_default_']) if isinstance(args['_default_'], str) \
                else {**args.pop('_default_'), **args}

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
        *root, file, _ = path.replace('.', '/').rsplit('/', 2)
        root = root[0].strip('/') + '/' if root else ''
        for base in module_paths:
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
        if isinstance(value, (Args, dict)) and key in original and isinstance(original[key], (Args, dict)):
            original[key].update(recursive_update(original[key], value))
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

    if args is None:
        args = arg

    def _interpolate(match_obj):
        if match_obj.group() is not None:
            try:
                got = get(args, match_obj.group()[2:][:-1])
                # TODO Try this for classes and types
                # out = got.__name__ if isinstance(got, type) else type(got).__name__ if str(got)[:10] == '<__main__.' \
                #     else str(got)
                out = str(got)
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
