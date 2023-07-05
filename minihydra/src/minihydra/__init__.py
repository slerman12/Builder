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

# minihydra.yaml_search_paths.append(path)
yaml_search_paths = [app, os.getcwd(), './', '/']  # List of paths

for path in yaml_search_paths:
    if path not in sys.path:
        sys.path.append(path)

added_modules = {}


# Simple-sophisticated instantiation of a class or module by various semantics
def instantiate2(args, i=0, **kwargs):
    if isinstance(args, (Args, dict)):
        args = Args(args)  # Non-destructive shallow copy

    if hasattr(args, '_override_'):
        kwargs.update(args.pop('_override_'))  # For loading old models with new, overridden args

    while hasattr(args, '_default_'):  # Allow inheritance between shorthands
        if isinstance(args['_default_'], str):
            args = args['_default_']
        else:
            args_ = args.pop('_default_')
            args_.update(args)
            args = args_

    if hasattr(args, '_target_') and args._target_:
        if isinstance(args._target_, nn.Module):  # Allow objects as _target_
            args = args._target_
        else:
            try:
                return minihydra.instantiate(args, **kwargs)  # Regular hydra
            except ImportError as e:
                if '(' in args._target_ and ')' in args._target_:  # Direct code execution
                    args = args._target_
                else:
                    if 'Utils.' in args._target_:
                        raise ImportError
                    args._target_ = 'Utils.' + args._target_  # Portal into Utils
                    try:
                        return instantiate(args, i, **kwargs)
                    except ImportError:
                        raise e  # Original error if all that doesn't work
            except TypeError as e:
                kwarg = re.search('got an unexpected keyword argument \'(.+?)\'', str(e))
                if kwarg and kwarg.group(1) not in args:
                    kwargs = {key: kwargs[key] for key in kwargs if key != kwarg.group(1)}
                    return instantiate(args, i, **kwargs)  # Signature matching, only for kwargs not args
                raise e  # Original error

    if isinstance(args, str):
        for key in kwargs:
            args = args.replace(f'kwargs.{key}', f'kwargs["{key}"]')  # Interpolation
        args = eval(args)  # Direct code execution

    # Signature matching
    if isinstance(args, type):
        _args = signature(args).parameters
        args = args(**kwargs if 'kwargs' in _args else {key: kwargs[key] for key in kwargs.keys() & _args})

    return None if hasattr(args, '_target_') \
        else args[i] if isinstance(args, (list, nn.ModuleList)) \
        else args  # Additional useful ones


"""
Instantiate plans

Accepts a config or string or None or object, and kwargs.
If string, create config if path, else execute with kwargs interpolation.
If config, instantiate. If _target_ None, return None.
If None, return None.
If object callable, instantiate with kwargs and optionally signature matching. Else return object. If iterable, index.

Path belongs to config _target_.
Path can be dot-separated format w.r.t. search paths, including "..".
Path can be relative or absolute directory path.
Path can be dot-separated format w.r.t. modules, defined locally or globally, or module search paths e.g. "Utils.".

Everything in UnifiedML instantiate: _default_, _override_, optionally: signature matching, support for objects, funcs.
    Or better, allow adding rules.
"""


# Something like this
def instantiate(args, **kwargs):
    if args is None:
        return

    if isinstance(args, str):
        args = Args(_target_=args)

    # args = recursive_Args(args)  # Why does it need to make a copy?
    args = Args(args)
    args.update(kwargs)

    file, *module = args.pop('_target_').rsplit('.', 1)

    sub_module, *sub_modules = file.split('.')

    # Can instantiate based on added modules
    if sub_module in added_modules:
        sub_module = added_modules[sub_module]

        try:
            for key in sub_modules + module:
                sub_module = getattr(sub_module, key)

            return sub_module(**args)
        except AttributeError:
            pass

    # file = file.replace('.', '/').replace('.py', '')  # TODO: Can it search wrt absolute paths?
    file = file.replace('..', '$#').replace('.', '/').replace('$#', '..').replace('.py', '')
    if module:
        module = module[0]
    else:
        module = file
        file = 'Utils'  # TODO Generalize this / allow added modules
    for i, path in enumerate(yaml_search_paths):
        for j, file in enumerate([file + '/__init__', file]):
            if not os.path.exists(path + '/' + file + '.py'):
                if i == len(yaml_search_paths) - 1 and j == 1:
                    raise FileNotFoundError(f'Could not find {module} in /{file}.py. '
                                            f'Search paths include: {yaml_search_paths}')
                continue

            # Reuse cached imports
            if file.replace('/', '.').replace('...', '..') + '_inst' in sys.modules:
                module = getattr(sys.modules[file.replace('/', '.').replace('...', '..') + '_inst'], module)
                return module(**args) if callable(module) else module

            # Reuse cached imports
            for key, value in sys.modules.items():
                if hasattr(value, '__file__') and value.__file__ and path + '/' + file + '.py' in value.__file__:
                    try:
                        module = getattr(value, module)
                        return module(**args) if callable(module) else module
                    except AttributeError:
                        if i == len(yaml_search_paths) - 1 and j == 1:
                            raise AttributeError(f'Could not initialize {module} in /{file}.py.')
                        continue

            # Import
            package = importlib.import_module(file.replace('/', '.').replace('...', '..'))
            sys.modules[file.replace('/', '.').replace('...', '..') + '_inst'] = package
            module = getattr(package, module)
            return module(**args) if callable(module) else module


def open_yaml(source):
    for path in yaml_search_paths + ['']:
        try:
            with open(path + '/' + source.strip('/'), 'r') as file:
                args = yaml.safe_load(file)
            return recursive_Args(args)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f'{source} not found. Searched: {yaml_search_paths + [""]}')


class Args(dict):
    def __init__(self, _dict=None, **kwargs):
        super().__init__()
        self.__dict__ = self  # Allows access via attributes
        self.update({**(_dict or {}), **kwargs})

    def to_dict(self):
        return {**{key: self[key].to_dict() if isinstance(self[key], Args) else self[key] for key in self}}


# Allow access via attributes recursively
def recursive_Args(args):
    if isinstance(args, dict):
        args = Args(args)

    items = enumerate(args) if isinstance(args, list) \
        else args.items() if isinstance(args, Args) else ()  # Iterate through lists, tuples, or dicts

    for key, value in items:
        args[key] = _parse(recursive_Args(value))  # Recurse through inner values

    return args


def recursive_update(args, args2):
    for key, value in args2.items():
        if isinstance(value, type(args2)) and key in args:
            args[key].update(recursive_update(args.get(key, {}), value))
        else:
            args[key] = value
    return args


def read(source, parse_task=True):
    args = open_yaml(source)

    # Need to allow imports  TODO Might have to add to yaml_search_paths
    if 'imports' in args:
        imports = args.pop('imports')

        self = recursive_Args(args)

        for module in imports:
            module = self if module == 'self' else read(module + '.yaml', parse_task=False)
            recursive_update(args, module)

    # Parse task  TODO Save these in minihydra: log_dir:
    if parse_task:  # Not needed in imports recursions
        for sys_arg in sys.argv[1:]:
            key, value = sys_arg.split('=', 1)
            if key == 'task':
                args['task'] = value

        # Command-line task import
        if 'task' in args and args.task not in [None, 'null']:
            try:
                task = read('task/' + args.task + '.yaml', parse_task=False)
            except FileNotFoundError:
                task = read(args.task + '.yaml', parse_task=False)
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
    # Parse command-line  TODO Save these in minihydra: log_dir:
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


def multirun(args):
    # Divide args into multiple copies
    return args


# Can just get args, no decorator
def just_args(source=None):
    if source is not None:
        yaml_search_paths.append(app + '/' + source.split('/', 1)[0])

    args = Args() if source is None else read(source)
    args = parse(args)
    args = interpolate(args)  # Command-line requires quotes for interpolation
    # args = multirun(args)

    return args


# Can decorate a method with args in signature
def get_args(source=None):
    def decorator_func(func):
        return lambda: func(just_args(source))
    return decorator_func
