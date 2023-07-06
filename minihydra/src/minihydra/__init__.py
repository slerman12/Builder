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

sys.path.extend([app, os.getcwd()])  # Adding sys paths instead of module paths so that imports in modules work as well

yaml_search_paths = [app, os.getcwd()]  # List of paths to search for yamls in
module_paths = [app, os.getcwd()]  # List of paths to instantiate modules from
added_modules = {}  # Name: module pairs to instantiate from


def get_module(_target_, paths=None, modules=None):
    paths = list(paths or []) + module_paths

    if modules is None:
        modules = Args()

    modules.update(added_modules)

    if '/' in _target_:
        if '.py.' in _target_:
            path, module_name = _target_.split('.py.', 1)
            path += '.py'
        else:
            if '.py' in _target_:
                path, module_name = _target_, None
            else:
                assert '.' in _target_, f'Directory path must include a .<module-name>, got: {_target_}'
                path, module_name = _target_.rsplit('.', 1)
                path += '/__init__.py.'
    else:
        *path, module_name = _target_.rsplit('.', 1)
        path = path[0].replace('..', '!@#$%^&*').replace('.', '/').replace('!@#$%^&*', '../') + '.py' if path else None

    if path:
        *prefix, path = path.rsplit('../', 1)
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
                if prefix:
                    base += prefix[0] + '../'  # Move relative backwards to base
                base = os.path.abspath(base)
                if base:
                    base += '/'

                if not os.path.exists(base + path):
                    if os.path.exists(base + path.replace('.py', '/__init__.py')):
                        path = path.replace('.py', '/__init__.py')
                    else:
                        continue

                # # Start from the absolute path, adding only the highest-level necessary path to the system path
                # parts = [part.replace('.py', '') for part in os.path.abspath(base + path).split('/') if part]
                # name = added = ''
                # explored = '/'
                #
                # # Construct a module path name given a directory path
                # for part in parts:
                #     name += '.' + part if name else part
                #     if not os.path.exists(name.replace('.', '/') + '.py'):
                #         if added:
                #             sys.path.pop(sys.path.index(added))
                #             added = ''
                #         if explored not in sys.path:
                #             sys.path.append(explored)  # Add exactly 0 or 1 sys paths
                #             added = explored
                #         name = part
                #     explored += '/' + part if explored else part

                # Check if cached
                # if name in sys.modules:
                #     module = sys.modules[name]
                # else:
                    # for key, value in sys.modules.items():
                    #     if hasattr(value, '__file__') and value.__file__ and base + path in value.__file__:
                    #         module = value
                    #         break
                    # else:
                    #     # Finally, import
                    #     module = importlib.import_module(name)
                    #     sys.modules[name] = module

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
                            add, path = path.rsplit('.', 1)
                            sys.path.append(base + add)
                            module = importlib.import_module(path)
                            path = add + '.' + path
                        sys.modules[path] = module
                        break
        if module is None:
            raise FileNotFoundError(f'Could note find path {path}. Search paths include: {paths}')
        else:
            # Return the relevant module
            return module if module_name is None else getattr(module, module_name)
    elif module_name in modules:
        # Return the module from already-defined modules
        return modules[module_name]
    else:
        for module in modules.values():
            if hasattr(module, module_name):
                return getattr(module, module_name)
    raise FileNotFoundError(f'Could note find module {module_name}. Search modules include: {list(modules.keys())}')


def instantiate(args, _i_=None, _paths_=None, _modules_=None, _signature_matching_=True, **kwargs):
    if hasattr(args, '_target_') or hasattr(args, '_default_'):
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
                args = args.replace(f'kwargs.{key}', f'kwargs["{key}"]')  # Interpolation
            module = eval(_target_, globals(), {**added_modules, **(_modules_ or {})})  # Direct code execution
        else:
            module = _target_

            if isinstance(module, str):
                module = get_module(module, _paths_, _modules_)

            if callable(module):
                # Signature match, only for kwargs not args
                _args = inspect.signature(module).parameters if _signature_matching_ else kwargs.keys()
                args.update(kwargs if 'kwargs' in _args else {key: kwargs[key] for key in kwargs.keys() & _args})
                module = module(**args)
    else:
        # Convert to config
        return instantiate(Args(_target_=args), _i_, _paths_, _modules_, _signature_matching_, **kwargs)

    try:
        iter(module)
    except TypeError:
        return module
    else:
        return module if _i_ is None else module[_i_]  # Allow sub-indexing (if specified)


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

    # Need to allow imports  TODO Might have to add relative paths to yaml_search_paths !
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
