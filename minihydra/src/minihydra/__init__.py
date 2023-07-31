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
import os.path
import re
import sys
from math import inf
import yaml

app = '/'.join(str(inspect.stack()[-1][1]).split('/')[:-1])
cwd = os.getcwd()

sys.path.extend([app, cwd])  # Adding sys paths instead of module paths so that imports in modules work as well

yaml_search_paths = [app, cwd]  # List of paths to search for yamls in
module_paths = [app, cwd]  # List of paths to instantiate modules from
added_modules = {}  # Name: module pairs to instantiate from

log_dir = None


def get_module(_target_, paths=None, modules=None):
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


def instantiate(args, _i_=None, _paths_=None, _modules_=None, _signature_matching_=True, _verbose_=False, **kwargs):
    if hasattr(args, '_target_') or hasattr(args, '_default_') or \
            isinstance(args, dict) and ('_target_' in args or '_default_' in args):
        args = Args(args)

        if '_override_' in args:
            kwargs.update(args.pop('_override_'))  # For overriding args without modifying defaults

        while '_default_' in args:  # Allow inheritance between sub-args Note: For some reason 2nd-last is dict not Args
            args = Args(_target_=args['_default_']) if isinstance(args['_default_'], str) \
                else {**args.pop('_default_'), **args}

        _target_ = args.pop('_target_')

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
                if _verbose_:
                    print('instantiated', _target_)
                module = module(**args)
    else:
        # Convert to config
        return instantiate(Args(_target_=args), _i_, _paths_, _modules_, _signature_matching_, **kwargs)

    # Allow sub-indexing (if specified)
    return module[_i_] if (isinstance(module, (list, tuple)) or 'ModuleList' in str(type(module))) and _i_ is not None \
        else module


def open_yaml(source, return_path=False):
    for path in yaml_search_paths + ['']:
        try:
            with open(path + '/' + source.strip('/'), 'r') as file:
                args = yaml.safe_load(file)
            return (recursive_Args(args), path + '/' + source.strip('/')) if return_path else recursive_Args(args)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f'{source} not found. Searched: {yaml_search_paths + [""]}')


class Args:
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

    def get(self, key, *__default):
        return self.__dict__.get(key, *__default)

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


def recursive_update(original, update):
    for key, value in update.items():
        if isinstance(value, (Args, dict)) and key in original and isinstance(original[key], (Args, dict)):
            original[key].update(recursive_update(original[key], value))
        else:
            original[key] = value
    return original


def read(source, recurse=False):
    args, path = open_yaml(source, return_path=True)

    # Need to allow imports
    if 'imports' in args:
        imports = args.pop('imports')

        self = recursive_Args(args)

        added = None
        for module in imports:
            path = os.path.dirname(path)
            if path not in sys.path:
                added = path
                yaml_search_paths.append(path)
            try:
                module = self if module == 'self' else read(module + '.yaml', recurse=True)
            except FileNotFoundError as e:
                try:
                    module = read('task/' + module + '.yaml', recurse=True)
                except FileNotFoundError:
                    raise e
            if added:
                yaml_search_paths.pop(yaml_search_paths.index(added))
                added = None
            recursive_update(args, module)

    # Parse task
    if not recurse:
        for sys_arg in sys.argv[1:]:
            key, value = sys_arg.split('=', 1)
            if key == 'task':
                args['task'] = value

    # Command-line task import
    if 'task' in args and args.task not in [None, 'null']:
        added = None
        path = os.path.dirname(path)
        if path not in sys.path:
            added = path
            yaml_search_paths.append(path)
        try:
            task = read('task/' + args.task + '.yaml', recurse=True)
        except FileNotFoundError as e:
            try:
                task = read(args.task + '.yaml', recurse=True)
            except FileNotFoundError:
                raise e
        if added:
            yaml_search_paths.pop(yaml_search_paths.index(added))
        recursive_update(args, task)

    return args


def _parse(value):
    if isinstance(value, str):
        if re.compile(r'^\[.*\]$').match(value) or re.compile(r'^\{.*\}$').match(value) or \
                re.compile(r'^-?[0-9]*.?[0-9]+(e-?[0-9]*.?[0-9]+)?$').match(value):
            value = ast.literal_eval(value)  # TODO Custom with no quotes required for strings
        elif isinstance(value, str) and value.lower() in ['true', 'false', 'null', 'inf']:
            value = True if value.lower() == 'true' else False if value.lower() == 'false' \
                else None if value.lower() == 'null' else inf
    return value


def parse(args=None):
    # Parse command-line
    for sys_arg in sys.argv[1:]:
        arg = args
        keys, value = sys_arg.split('=', 1)
        keys = keys.split('.')
        for key in keys[:-1]:
            if key not in arg:
                setattr(arg, key, Args())
            arg = getattr(arg, key)
        setattr(arg, keys[-1], value)
        arg[keys[-1]] = _parse(value)
    return args


def get(args, keys):
    arg = args
    keys = keys.split('.')
    for key in keys:
        arg = getattr(arg, key)
    return interpolate([arg], args)[0]  # Interpolate to make sure gotten value is resolved


# minihydra.grammar.append(rule)
grammar = []  # List of funcs


def interpolate(arg, args=None):
    if args is None:
        args = arg

    def _interpolate(match_obj):
        if match_obj.group() is not None:
            try:
                out = str(get(args, match_obj.group()[2:][:-1]))
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


def log(args):
    if 'minihydra' in args:
        if 'log_dir' in args.minihydra:
            path = interpolate([args.minihydra.log_dir], args)[0] + '.yaml'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as file:
                args = interpolate(parse(Args()), args)
                args.update(_minihydra_={'app': app, 'cwd': cwd})
                yaml.dump(args.to_dict(), file, sort_keys=False)


def multirun(args):
    # Divide args into multiple copies
    pass


# Can just get args, no decorator
def just_args(source=None, logging=False):
    if source is not None:
        yaml_search_paths.append(app + '/' + source.split('/', 1)[0])

    args = Args() if source is None else read(source)
    args = parse(args)
    args = interpolate(args)  # Command-line requires quotes for interpolation
    if logging:
        log(args)

    return args


# Can decorate a method with args in signature
def get_args(source=None, logging=True):
    def decorator_func(func):
        return lambda: func(just_args(source, logging=logging))

    return decorator_func
