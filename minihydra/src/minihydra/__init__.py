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
from importlib.machinery import SourceFileLoader
from math import inf
import yaml


app = '/'.join(str(inspect.stack()[-1][1]).split('/')[:-1])

yaml_search_paths = [app, os.getcwd()]  # List of paths to search for yamls in
module_paths = [app, os.getcwd()]  # List of paths to instantiate modules from
added_modules = {}  # Name: module pairs to instantiate from


"""
Minihydra plans

Accepts a config or string or None or object, and kwargs.
If None, return None.
If string, create config if path, else (if parens) execute with kwargs interpolation, optionally signature matching, 
    and modules as locals.
If config, instantiate. If _target_ None, return None.
If object callable, instantiate with kwargs and optionally signature matching. Else return object. If iterable, index.

Path belongs to config _target_.
Path can be dot-separated format w.r.t. search paths, including "..".
Path can be relative or absolute directory path.
Path can be dot-separated format w.r.t. modules, defined locally or globally, or module search paths e.g. "Utils.".
    Perhaps then no need for sys.path.append anywhere. yaml_search_paths, module_paths, modules.

Everything in UnifiedML instantiate: _default_, _override_, optionally: signature matching, support for objects, funcs.
    Or better, allow adding rules (funcs based on args, kwargs that return args, kwargs). Can skip & remove _override_.
    
Also, Utils can manually map Uppercase to existing lowercases-with-_target_ attr. 
    Or even create sub-configs for some e.g.senses.Poo creates a new senses.poo={_target_: poo}.
As well as constructing "recipes" from the main shorthands.

minihydra can have a pseudonyms arg with main_name: pseudonyms-list sublists maybe. Instead of _default_.

minihydra can allow adding yaml_search_paths, module_paths, and modules via command line as reserved arguments. 
    Maybe add underscores to all reserved arguments.
    
Maybe add __file__ directly from get_args call to add paths/modules.
"""


def get_module1(args):
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


def get_module(_target_, paths=None, modules=None):
    paths = list(paths or []) + module_paths

    if modules is None:
        modules = Args()

    modules.update(added_modules)

    if '/' in _target_:
        if '.py.' in _target_:
            path, module_name = _target_.split('.py.')
            path += '.py.'
        else:
            path, module_name = _target_, None
            assert '.py' in path, f'Directory path must include a .py file, got: {path}'
    else:
        *path, module_name = _target_.rsplit('.', 1)
        path = path[0].replace('..', '!@#$%^&*').replace('.', '/').replace('!@#$%^&*', '../') + '.py' if path else None

    if path:
        keys = path.split('/')
        module = None

        if keys and keys[0] in modules:
            module = modules[keys[0]]

            for key in keys[1:]:
                module = getattr(module, key.replace('.py', ''))
        else:
            for i, base in enumerate(paths + ['']):
                if base and base[-1] != '/':
                    base += '/'

                if not os.path.exists(base + path):
                    continue

                # name = (base + path).split('.py', 1)[0].replace('/', '.').replace('...', '..').strip('.')
                # module_spec = importlib.util.spec_from_file_location(name, base + path)
                # module = importlib.util.module_from_spec(module_spec)
                # sys.modules[name] = module
                # module_spec.loader.exec_module(module)  # Import


                # relpath = os.path.relpath(base + path)
                # top_level = ''.join(['../'] * relpath.count('../'))
                # if top_level not in sys.path:
                #     sys.path.append(top_level)
                added = module = None
                parts = os.path.abspath(base + path).split('/')
                for j, part in enumerate(parts):
                    if part:
                        if module is not None:
                            module = getattr(module, part.replace('.py', ''), None)
                        if module is None:
                            if added:
                                sys.path.pop(sys.path.index(added))
                                added = None
                            add = '/'.join(parts[:j])
                            if add not in sys.path:
                                sys.path.append(add)
                                added = add
                            try:
                                module = __import__(part.replace('.py', ''))
                            except ModuleNotFoundError:
                                pass
                # name = '.'.join(name).replace('.py', '')
                # print(name)
                # module = __import__(name)

                # importable_path = '/'.join([part for part in abspath.split('/')])
                # name = os.path.relpath(base + path).replace('../', '')
                # module_spec = importlib.util.spec_from_file_location(base + path, base + path)
                # module = importlib.util.module_from_spec(module_spec)
                # sys.modules[base + path] = module
                # module_spec.loader.exec_module(module)  # Import

                # Import
                # module = __import__(os.path.basename(path).replace('.py', ''))
                # sys.modules[os.path.basename(path).replace('.py', '')] = module
                # module = importlib.import_module(os.path.basename(path).replace('.py', ''))
                # print(os.path.basename(path).replace('.py', ''))
                # sys.path.pop()
                # module = SourceFileLoader(base + path, base + path).load_module()  # TODO Verify cached
                # sys.modules[base + path] = module
        if module is None:
            raise FileNotFoundError(f'Could note find path {path}. Search paths include: {paths}')
        else:
            return module if module_name is None else getattr(module, module_name)
    elif module_name in modules:
        return modules[module_name]
    else:
        raise FileNotFoundError(f'Could note find module {module_name}. Search modules include: {list(modules.keys())}')


def instantiate(args, i=None, paths=None, modules=None, signature_matching=True, **kwargs):
    if hasattr(args, '_target_'):
        args = Args(args)
        _target_ = args.pop('_target_')

        if _target_ is None:
            return
        elif isinstance(_target_, str) and '(' in _target_ and ')' in _target_:  # Function calls
            for key in kwargs:
                args = args.replace(f'kwargs.{key}', f'kwargs["{key}"]')  # Interpolation
            locals().update(modules or {})
            module = eval(_target_, locals())  # Direct code execution
        else:
            module = _target_

            if isinstance(module, str):
                module = get_module(module, paths, modules)

            if callable(module):
                # Signature match, only for kwargs not args
                _args = inspect.signature(module).parameters if signature_matching else kwargs.keys()
                args.update(kwargs if 'kwargs' in _args else {key: kwargs[key] for key in kwargs.keys() & _args})
                module = module(**args)
    else:
        # Convert to config
        return instantiate(Args(_target_=args), i, paths, modules, signature_matching, **kwargs)

    try:
        iter(module)
    except TypeError:
        return module
    else:
        return module if i is None else module[i]  # Allow sub-indexing (if specified)


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

    # Need to allow imports  TODO Might have to add relative paths to yaml_search_paths
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
