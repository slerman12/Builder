# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import ast
import inspect
import math
import textwrap
import time
import os
import sys
import random
from copy import deepcopy
from functools import cached_property
import re
from multiprocessing.pool import ThreadPool

from torch.nn import Identity, Flatten  # For direct accessibility via command line
import torchvision  # For direct accessibility via command line
from torchvision import transforms  # For direct accessibility via command line
from Agents.Blocks.Augmentations import *  # For direct accessibility via command line
from Agents.Blocks.Architectures import *  # For direct accessibility via command line

import warnings

import numpy as np

import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *

from minihydra import Args, yaml_search_paths, module_paths, added_modules, grammar, instantiate, interpolate, \
    get_module


# Sets all Pytorch and Numpy random seeds
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Initializes seeds, device, and CUDA acceleration
def init(args):
    # Customize working dir
    if args.path:
        os.chdir(args.path)

        if args.path not in yaml_search_paths:
            yaml_search_paths.append(args.path)

        if args.path + '/Hyperparams' not in yaml_search_paths and os.path.exists(args.path + '/Hyperparams'):
            yaml_search_paths.append(args.path + '/Hyperparams')

    # For launching via an external app
    args.update(launch_args)

    # Set seeds
    set_seeds(args.seed)

    mps = getattr(torch.backends, 'mps', None)  # M1 MacBook speedup

    # Set device
    args.device = args.device if args.device != '???' else 'cuda' if torch.cuda.is_available() \
        else 'mps' if mps and mps.is_available() \
        else 'cpu'

    # CUDA speedup via automatic mixed precision
    MP.enable(args)

    # CUDA speedup when input sizes don't vary
    torch.backends.cudnn.benchmark = True

    print('Device:', args.device)

    # Set passed-in model in agent
    preconstruct_agent(args.agent, args.model)

    interpolate(args)


UnifiedML = os.path.dirname(__file__)
app = '/'.join(str(inspect.stack()[-1][1]).split('/')[:-1])


# Imports UnifiedML paths and the paths of any launching app
def import_paths():
    if UnifiedML not in yaml_search_paths:
        yaml_search_paths.append(UnifiedML)  # Adds UnifiedML to yaml search path

    if UnifiedML not in module_paths:
        module_paths.append(UnifiedML)  # Adds UnifiedML to module instantiation search path

    added_modules.update(globals())  # Adds everything in Utils to module instantiation path TODO Manually specify

    # Adds Hyperparams dir to search path
    for path in [UnifiedML, app, os.getcwd()]:
        if path + '/Hyperparams' not in yaml_search_paths and os.path.exists(path + '/Hyperparams'):
            yaml_search_paths.append(path + '/Hyperparams')


import_paths()


# Grammar rules for minihydra
def parse(arg, key, func, resolve=lambda name: name):
    pattern = r'\$\{' + key + r':([^(\$\{|\})]+)\}'
    if isinstance(arg, str) and re.match(r'.*' + pattern + r'.*', arg):
        return resolve(re.sub(pattern, lambda name: func(re.findall(pattern, name.group())[0]), arg))
    return arg


def grammars():
    # Format path names
    # e.g. "Checkpoints/Agents.DQNAgent" -> "Checkpoints/DQNAgent"
    grammar.append(lambda arg: parse(arg, 'format', lambda name: name.split('.')[-1]))

    # A boolean "not" operation for config
    grammar.append(lambda arg: parse(arg, 'not', lambda bool: str(not ast.literal_eval(bool)),
                                     lambda name: ast.literal_eval(name)))


grammars()

launch_args = {}


# Launches UnifiedML from inside a launching app with specified args
def launch(**args):
    from Run import main

    original = list(sys.argv)

    command_line_args = {arg.split('=')[0] for arg in sys.argv if '=' in arg}
    added = set()

    for key, value in args.items():
        if isinstance(value, (str, bool)):
            if key not in command_line_args:
                sys.argv.append(f'{key}={value}')  # For minihydra grammars in Utils  TODO Maybe just interpolate
                added.add(key)

    global launch_args
    launch_args = {key: args[key] for key in args.keys() - command_line_args - added}

    main()  # Run

    launch_args = {}
    sys.argv = original


# Saves model + args + selected attributes
def save(path, model, args=None, *attributes):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({'state_dict': model.state_dict(),
                'attr': {attr: getattr(model, attr) for attr in attributes},
                'args': args}, path)
    print(f'Model successfully saved to {path}')


# Loads model or part of model
def load(path, device='cuda', args=None, preserve=(), distributed=False, attr='', **kwargs):
    while True:
        try:
            to_load = torch.load(path, map_location=device)  # Load
            original_args = to_load['args']

            args = args or original_args or {}  # Note: Could instead use original_args, but support _overload_
            if 'obs_spec' in original_args:
                args['obs_spec'] = original_args['obs_spec']  # Since norm and standardize stats may change
            if 'recipes' in original_args:
                args['recipes'] = original_args['recipes']  # Since assumed TODO Default to these where current are null
            break
        except Exception as e:  # Pytorch's load and save are not atomic transactions, can conflict in distributed setup
            if not distributed:
                raise RuntimeError(e)
            warnings.warn(f'Load conflict, resolving...')  # For distributed training

    # Overriding original args where specified
    for key, value in kwargs.items():  # Need to update kwargs to include _overload_?
        if attr:
            args.recipes[attr + f'._overload_.{key}'] = value
        else:
            args[key] = value

    model = instantiate(args).to(device)

    # Load model's params
    model.load_state_dict(to_load['state_dict'], strict=False)
    model.device = device

    # Load saved attributes as well
    for key, value in to_load['attr'].items():
        if hasattr(model, key) and key not in ['state_dict', *preserve]:
            setattr(model, key, value)

    # Can also load part of a model. Useful for recipes,
    # e.g. python Run.py Eyes=load +eyes.path=<Path To Agent Checkpoint> +eyes.attr=encoder.Eyes +eyes.device=<device>
    for key in attr.split('.'):
        if key:
            model = getattr(model, key)
    print(f'Successfully loaded {attr if attr else "agent"} from {path}')
    return model


# TODO:
#     What if type method rather than object method? Would model have to be passed in for 'self'? Does/can this happen?
#     What if parallel?
#     args.agent_name = model
# Depending on what is passed in, different agent components can be defined
def preconstruct_agent(agent, model):
    if model._target_ is not None:
        _target_ = model._target_ if isinstance(model._target_, type) \
            else get_module(model._target_) if isinstance(model._target_, str) \
            else type(model._target_)

        signature = set(inspect.signature(_target_).parameters)

        # If model has act and learn methods, no shapes, and the learn method has no return statement, Agent <- Model
        ins = signature & {'in_shape', 'in_shape', 'in_dim', 'in_channels', 'in_features'}
        outs = signature & {'output_shape', 'out_shape', 'out_dim', 'out_channels', 'out_features'}
        if hasattr(_target_, 'act') and hasattr(_target_, 'learn') and not ins and not outs:
            # and not \
            # any(isinstance(node, ast.Return)
            #     for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(_target_.learn))))):  TODO Logger
            agent.update(model)
        else:
            if outs:
                agent.recipes.actor.Pi_head = _target_  # As Pi_head when output shape
                agent.recipes.encoder.Eyes = agent.recipes.encoder.pool = agent.recipes.actor.trunk = Identity()
            else:
                agent.recipes.encoder.Eyes = _target_  # Otherwise as Eyes

            # Override agent act/learn methods with model  Note: For-loop lambda breaks without the _key_= default
            for key in {'act', 'learn'}:
                if callable(getattr(_target_, key, ())) and \
                        ('_overrides_' not in agent or key not in agent._overrides_):
                    agent.setdefault('_overrides_', Args())[key] = lambda a, *v, _key_=key, **k: getattr(
                        a.encoder.Eyes if not outs
                        else a.actor.Pi_head if agent.num_actors > 1 else a.actor.Pi_head.ensemble[0], _key_)(*v, **k)

        def preconstruct_optimize(loss_fn):
            def overriden(a, replay):
                loss = loss_fn(a, replay)
                # etc.

            return overriden

        # if 'learn' in agent._overrides_ and agent._overrides_.learn is not None:  TODO Logger
        #     _target_ = get_module(agent._overrides_.learn)
        #
        #     # Logger optional
        #     if len(inspect.signature(_target_).parameters) == 2:
        #         agent._overrides_.learn = lambda a, replay, None: _target_(a, replay)
        #
        #     # If learn has a return statement
        #     if any(isinstance(node, ast.Return)
        #            for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(_target_.learn))))):
        #         # Treat as loss
        #         agent._overrides_.learn = preconstruct_optimize(_target_)

    # Agent infers a forward from act  TODO Why not define this in Agent?
    #  TODO Shouldn't instance type be returned by get_module
    _target_ = agent._target_ if isinstance(agent._target_, type) \
        else get_module(agent._target_) if isinstance(agent._target_, str) \
        else type(agent._target_)

    assert hasattr(_target_, 'act') and hasattr(_target_, 'learn'), 'Agent requires act & learn methods.'

    # Checks if Pytorch module has a forward user-defined
    if not any(isinstance(node, ast.Return)
               for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(_target_.forward))))):
        if '_overrides_' not in agent or 'forward' not in agent._overrides_:
            agent.setdefault('_overrides_', Args())['forward'] = lambda a, *v, **k: \
                a.act(*v, **k)[0].squeeze(1).squeeze(-1)

    # Logger optional
    # if len(inspect.signature(_target_.learn).parameters) == 2:
    #     agent.setdefault('_overrides_', Args())['learn'] = lambda a, replay, logger: _target_.learn(a, replay)


class MultiModal(nn.Module):
    def __init__(self, args=None, **parts):
        super().__init__()

        args = args or Args()

        # Default accepted mappings
        defaults = Args(Eyes={'obs', 'image', 'sight'}, Ears={'audio', 'sound'}, Proprio={'features', 'touch'})

        # If args is itself a part, then apply it to (Eyes, Obs, Image, obs, image). a.k.a. Eyes is default name.
        # Note. Maybe just do all lowercase.
        # If args is string, instantiate-it.

        # Maps parts names (uppercase) to corresponding datums (lowercase)
        #   s.t. if datums are present, they get fed as input to the corresponding part
        self.datums = Args()

        # Synonymous datums
        for name in args.keys() | parts.keys():
            # All parts can accept datums with matching-key name
            self.datums[name.upper()] = {name.lower()} | set(n.lower() for n in args[name].pop('datums', ()))
            if name.upper() in defaults:
                self.datums[name.upper()] = self.datums[name.upper()] | defaults[name.upper()]

        # Default possible batch datums
        batch_keys = {'obs', 'action', 'label', 'reward'} | set().union(self.datums.values())

        self.datums.update({key.upper(): key for key in batch_keys if key.upper() not in self.datums})

        # Convert and pop Uppercase to _target_ syntax if str and not already

        # Passed-in modules
        self.parts.update({name.upper(): part for name, part in parts.items() if part is not None})
        # Instantiate
        self.parts.update({name.upper(): instantiate(args[name]) for name in args if name.upper() not in self.parts and
                           '_target_' in args[name]})
        # Default batch keys and specified datums
        self.parts = Args({key.upper(): nn.Identity() for key in batch_keys if key.upper() not in self.parts})

    def forward(self, batch):
        # Parts collect their batch items
        #   If none present, don't include part
        # Output with lowercase version of part name
        # If input isn't a batch, treats it as obs, returns non-batch.
        pass


# Launches and synchronizes multiple tasks
class MultiTask:
    def __init__(self):
        self.multi_task_enabled = False
        self.num_tasks = 0
        self.agents = []
        # TODO Support unify=False to just effectively multirun
        self.union = {'encoder': {'Eyes': [], 'pool': []},
                      'actor': {'trunk': [], 'Pi_head': []}, 'critic': {'trunk': [], 'Q_head': []}}
        self.unified_parts = set()
        self.synced = None

    def launch(self, multi_task):
        self.multi_task_enabled = True
        self.num_tasks = len(multi_task)

        if 'multi_task' in launch_args:
            launch_args.pop('multi_task')
        original_sys_args = sys.argv

        worker_start_times = [0.0] * self.num_tasks

        def create_thread(worker_task):
            worker, task = worker_task
            # Launch workers sequentially
            while worker and (time.time() - worker_start_times[worker - 1] < 1 or not worker_start_times[worker - 1]):
                time.sleep(1)

            worker_start_times[worker] = time.time()

            # With task-specific args
            task_args = [arg for arg in original_sys_args[1:-2] if arg.split('=')[0]
                         not in [task_arg.split('=')[0] for task_arg in task.split()] + ['multi_task']] + task.split()
            sys.argv = [sys.argv[0], *task_args, *sys.argv[-2:]]
            launch(**launch_args)  # Run

        print(f'Launching {self.num_tasks} tasks among Unified Agents!')

        with ThreadPool() as p:
            p.map(create_thread, enumerate(multi_task))  # Launch multi-tasks

    # Share multi-task architecture parts across agents when unifiable (if part names match & state dicts are copyable)
    def unify_agent_models(self, agent, args, device, path=None):
        if self.multi_task_enabled:
            print(f'Unifying agent models across {self.num_tasks} tasks...')

            agent = self._unify_agent_models(agent, args, device, path)  # Unify model parts

            self.agents.append(agent)

            # When all agents unified, log to console
            if self.agents.index(agent) == 0:
                while len(self.agents) < self.num_tasks:
                    time.sleep(1)

                print('Done. ✓\nThe following model parts were successfully unified among agents: '
                      f'{", ".join(self.unified_parts)}.' if len(self.unified_parts) > 0
                      else 'Our analysis did not find unifiable parts between multi-task agents.')

            self.sync_blocking(agent)  # Thread-safe atomic learn operations

        return agent

    # Uniquely identify multi-task architecture parts and unify equivalents
    def _unify_agent_models(self, agent, args, device, path=None):
        # Iterate through all blocks
        for block in self.union:
            for part in self.union[block]:
                # Analogous parts check
                if hasattr(agent, block):
                    agent_part = getattr(getattr(agent, block), part)

                    if part in ['Pi_head', 'Q_head'] and isinstance(agent_part, Ensemble):
                        agent_part = agent_part.ensemble  # Instantiate ensembles from ModuleLists

                    # See if parameters can be copied over. If not, don't unify
                    unifiable = False

                    for unified_part in self.union[block][part]:  # TODO Recipe same ID check
                        # Same modules check
                        if [name for name, _ in agent_part.named_modules()] != \
                                [name for name, _ in unified_part.named_modules()]:
                            continue
                        # Compatible params check
                        try:
                            update_ema_target(unified_part, agent_part)
                            unifiable = unified_part  # Confirm unifiable
                            break
                        except RuntimeError:
                            continue

                    # Unify
                    if unifiable:
                        self.unified_parts.add(f'{block}.{part}')

                        # Re-instantiate/load agent with new recipe
                        setattr(getattr(args.recipes, block), part, unifiable)
                        agent = load(path, device, args) if path else instantiate(args).to(device)
                    else:
                        self.union[block][part].append(agent_part)  # ! Can also add recipe and do a recipe check above

        return agent

    # Prevent learning steps from conflicting across multi-task threads
    def sync_blocking(self, agent):
        if self.synced is None:
            self.synced = [False] * len(self.agents)
            self.synced[-1] = True

        learn = agent.learn

        def sync_block_learn(replay):
            self.block(agent)
            logs = learn(replay)
            self.sync(agent)

            return logs

        agent.learn = sync_block_learn

    # Prevent thread conflicts
    def block(self, agent):
        i = self.agents.index(agent)
        if i == 0:
            while not self.synced[-1]:
                time.sleep(0.1)
        else:
            while not self.synced[i - 1]:
                time.sleep(0.1)

    # Synchronize threads
    def sync(self, agent):
        i = self.agents.index(agent)
        if i == 0:
            self.synced[-1] = False
        else:
            self.synced[i - 1] = False
        self.synced[i] = True


MT = MultiTask()


# TODO Delete, just for MLP CNN on MNIST
# assert self.agents[0].actor.Pi_head == self.agents[1].actor.Pi_head
# assert torch.allclose(list(self.agents[0].actor.Pi_head.parameters())[0],
#                       list(self.agents[1].actor.Pi_head.parameters())[0])
# python XRD.py multi_task='["task=classify/mnist Trunk=Identity Predictor=MLP +predictor.depth=0 experiment=1", "task=classify/mnist Eyes=MLP experiment=2 +eyes.depth=0 Trunk=Identity Predictor=MLP +predictor.depth=0"]'
# python XRD.py multi_task='["task=classify/mnist experiment=1", "task=classify/mnist Eyes=MLP experiment=2 +eyes.depth=0"]'
# python XRD.py multi_task='["task=classify/mnist experiment=1", "task=classify/mnist Eyes=MLP experiment=2"]'
# python XRD.py multi_task='["task=classify/mnist", "task=classify/mnist Eyes=MLP"]'
# python XRD.py multi_task='["task=NPCNN Eyes=XRD.Eyes","task=NPCNN num_classes=230 Eyes=XRD.Eyes"]'
# python XRD.py multi_task='["task=NPCNN Eyes=XRD.Eyes","task=SCNN Eyes=XRD.Eyes"]'


# Adaptively fills shaping arguments in instantiated Pytorch modules
def adaptive_shaping(in_shape=None, out_shape=None):
    shaping = {}

    if in_shape is not None:
        if not isinstance(in_shape, (list, tuple)):
            in_shape = [in_shape]

        shaping.update(dict(input_shape=in_shape, in_shape=in_shape, in_dim=math.prod(in_shape),
                            in_channels=in_shape[0]))
        shaping['in_features'] = shaping['in_dim']

    if out_shape is not None:
        if not isinstance(out_shape, (list, tuple)):
            out_shape = [out_shape]

        shaping.update(dict(output_shape=out_shape, out_shape=out_shape, out_dim=math.prod(out_shape),
                            out_channels=out_shape[0]))
        shaping['out_features'] = shaping['out_dim']

    return shaping


# Initializes model weights a la orthogonal
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)) or isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose1d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# Initializes model optimizer. Default: AdamW
def optimizer_init(params, optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None):
    params = list(params)

    # Optimizer
    optim = len(params) > 0 and (instantiate(optim, params=params, lr=getattr(optim, 'lr', lr)) or lr
                                 and AdamW(params, lr=lr, weight_decay=weight_decay or 0))  # Default

    # Learning rate scheduler
    scheduler = optim and (instantiate(scheduler, optimizer=optim) or lr_decay_epochs
                           and CosineAnnealingLR(optim, lr_decay_epochs))  # Default

    return optim, scheduler


# Copies parameters from a source model to a target model, optionally EMA weighing
def update_ema_target(source, target, ema_decay=0):
    with torch.no_grad():
        for params, target_params in zip(source.state_dict().values(), target.state_dict().values()):
            target_params.copy_((1 - ema_decay) * params + ema_decay * target_params)


# Compute the output shape of a CNN layer
def cnn_layer_feature_shape(*spatial_shape, kernel_size=1, stride=1, padding=0, dilation=1):
    if padding == 'same':
        return spatial_shape
    axes = [size for size in spatial_shape if size]
    if type(kernel_size) is not tuple:
        kernel_size = [kernel_size] * len(axes)
    if type(stride) is not tuple:
        stride = [stride] * len(axes)
    if type(padding) is not tuple:
        padding = [padding] * len(axes)
    if type(dilation) is not tuple:
        dilation = [dilation] * len(axes)
    kernel_size = [min(size, kernel_size[i]) for i, size in enumerate(axes)]
    padding = [min(size, padding[i]) for i, size in enumerate(axes)]  # Assumes adaptive
    out_shape = [math.floor(((size + (2 * padding[i]) - (dilation[i] * (kernel_size[i] - 1)) - 1) / stride[i]) + 1)
                 for i, size in enumerate(axes)] + list(spatial_shape[len(axes):])
    return out_shape


# Compute the output shape of a whole CNN (or other architecture)
def cnn_feature_shape(chw, *blocks, verbose=False):
    channels, height, width = chw[0], chw[1] if len(chw) > 1 else None, chw[2] if len(chw) > 2 else None
    for block in blocks:
        if isinstance(block, (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d, nn.Conv1d, nn.AvgPool1d, nn.MaxPool1d)):
            channels = block.out_channels if hasattr(block, 'out_channels') else channels
            height, width = cnn_layer_feature_shape(height, width,
                                                    kernel_size=block.kernel_size,
                                                    stride=block.stride,
                                                    padding=block.padding)
        elif isinstance(block, nn.Linear):
            channels = block.out_features  # Assumes channels-last if linear
        elif isinstance(block, nn.Flatten) and (block.start_dim == -3 or block.start_dim == 1):
            channels, height, width = channels * (height or 1) * (width or 1), None, None  # Placeholder height/width
        elif isinstance(block, (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d)):
            size = to_tuple(block.output_size)  # Can be int
            pair = size[0] if isinstance(block, nn.AdaptiveAvgPool2d) else None
            height, width = (size[0], pair) if width is None else size + (pair,) * (2 - len(size))
        elif hasattr(block, 'repr_shape'):
            chw = block.repr_shape(*chw)
            channels, height, width = chw[0], chw[1] if len(chw) > 1 else None, chw[2] if len(chw) > 2 else None
        elif hasattr(block, 'modules'):
            for layer in block.children():
                chw = cnn_feature_shape(chw, layer, verbose=verbose)
                channels, height, width = chw[0], chw[1] if len(chw) > 1 else None, chw[2] if len(chw) > 2 else None
        if verbose:
            print(type(block), (channels, height, width))

    feature_shape = tuple(size for size in (channels, height, width) if size is not None)

    return feature_shape


# General-purpose shape pre-computation. Unlike above, uses manual forward pass through model(s).
def repr_shape(input_shape, *blocks):
    for block in blocks:
        input_shape = block(torch.ones(1, *input_shape)).shape[1:]
    return input_shape


# "Ensembles" (stacks) multiple modules' outputs
class Ensemble(nn.Module):
    def __init__(self, modules, dim=1):
        super().__init__()

        self.ensemble = nn.ModuleList([m if i == 0 or m != modules[i - 1]
                                       else deepcopy(m) for i, m in enumerate(modules)])
        self.dim = dim

        # TODO Not the most elegant. But works. Note: Using module[0] elsewhere might unexpectedly ensemble

        # This makes it possible to use model= syntax w ensembles
        if hasattr(modules[0], 'forward'):
            self.first = modules[0].forward

            if len(modules) > 1:
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


# Replaces tensor's batch items with Normal-sampled random latent
class Rand(nn.Module):
    def __init__(self, size=1, output_shape=None, uniform=False):
        super().__init__()

        self.output_shape = to_tuple(output_shape or size)
        self.uniform = uniform

    def repr_shape(self, *_):
        return self.output_shape

    def forward(self, *x):
        x = torch.randn((x[0].shape[0], *self.output_shape), device=x[0].device)

        if self.uniform:
            x.uniform_()

        return x


# (Multi-dim) one-hot encoding
def one_hot(x, num_classes, null_value=0, one_value=1):
    # assert x.shape[-1] == 1  # Can check this
    x = x.squeeze(-1).unsqueeze(-1)  # Or do this
    x = x.long()
    shape = x.shape[:-1]
    nulls = torch.full([*shape, num_classes], null_value, dtype=torch.float32, device=x.device)
    return nulls.scatter(len(shape), x, one_value).float()


# Differentiable one_hot via "re-parameterization"
def rone_hot(x, null_value=0):
    return x - (x - one_hot(torch.argmax(x, -1, keepdim=True), x.shape[-1]) * (1 - null_value) + null_value)


# Differentiable clamp via "re-parameterization"
def rclamp(x, min, max):
    return x - (x - torch.clamp(x, min, max))


# (Multi-dim) indexing
def gather(item, ind, dim=-1, ind_dim=-1):
    """
    Same as torch.gather, but automatically batches/broadcasts ind:
        item: [item.size(0), ..., item.size(N), item.size(dim), item.size(N + 2), ..., item.size(M)],
        ind: [item.size(i), ..., item.size(N), ind.size(ind_dim), item.size(N + 2), ..., item.size(j)] where i ≤ N j ≤ M
        --> [item.size(0), ..., item.size(N), ind.size(ind_dim), item.size(N + 2), ..., item.size(M)]
    """

    ind_shape = ind.shape[ind_dim:]  # ["ind_dim", ..., j]
    tail_shape = item.shape[dim:][len(ind_shape):]  # [j + 1, ..., M]

    ind = ind.long().expand(*item.shape[:dim], *ind_shape)  # Assumes ind.shape[ind_dim] is desired num indices
    ind = ind.reshape(ind.shape + (1,) * len(tail_shape)).expand(*ind.shape, *tail_shape)  # [0, ..., "ind_dim", ... M]

    return torch.gather(item, dim, ind)


# (Multi-dim) cartesian product
def batched_cartesian_prod(items: (list, tuple), dim=-1, collapse_dims=True):
    """
    # Get all combinations of tensors starting at "dim", keeping all dims before "dim" independent (as batches)

    # Given N tensors with leading dims (B1 x B2 x ... x BL),
    # a specified "dim" (can vary in size across tensors, e.g. D1, D2, ..., DN), "dim" = L + 1
    # and tail dims (O1 x O2 x ... x OT), returns:
    # --> cartesian prod, batches independent:
    # --> B1 x B2 x ... x BL x D1 * D2 * ... * DN x O1 X O2 x ... x OT x N
    # Or if not collapse_dims:
    # --> B1 x B2 x ... x BL x D1 x D2 x ... x DN x O1 X O2 x ... x OT x N

    Consistent with torch.cartesian_prod except generalized to multi-dim and batches.
    """

    lead_dims = items[0].shape[:dim]  # B1, B2, ..., BL
    dims = [item.shape[dim] for item in items]  # D1, D2, ..., DN
    tail_dims = items[0].shape[dim + 1:] if dim + 1 else []  # O1, O2, ..., OT

    return torch.stack([item.reshape(-1, *(1,) * i, item.shape[dim], *(1,) * (len(items) - i - 1), *tail_dims).expand(
        -1, *dims[:i], item.shape[dim], *dims[i + 1:], *tail_dims)
        for i, item in enumerate(items)], -1).view(*lead_dims, *[-1] if collapse_dims else dims, *tail_dims, len(items))


# Sequential of instantiations e.g. python Run.py Eyes=Sequential +eyes._targets_="[CNN, Transformer]"
class Sequential(nn.Module):
    def __init__(self, _targets_, i=0, **kwargs):
        super().__init__()

        self.Sequence = nn.ModuleList()

        for _target_ in _targets_:
            self.Sequence.append(instantiate(Args({'_target_': _target_}) if isinstance(_target_, str)
                                             else _target_, i, **kwargs))

            if 'input_shape' in kwargs:
                kwargs['input_shape'] = cnn_feature_shape(kwargs['input_shape'], self.Sequence[-1])

    def repr_shape(self, *_):
        return cnn_feature_shape(_, self.Sequence)

    def forward(self, obs, *context):
        out = (obs, *context)
        # Multi-input/output Sequential
        for i, module in enumerate(self.Sequence):
            out = module(*out)
            if not isinstance(out, tuple) and i < len(self.Sequence) - 1:
                out = (out,)
        return out


# Swaps image dims between channel-last and channel-first format (Convenient helper)
class ChannelSwap(nn.Module):
    def repr_shape(self, *_):
        return _[-1], *_[1:-1], _[0]

    def forward(self, x, spatial2d=True):
        return x.transpose(-1, -3 if spatial2d and len(x.shape) > 3 else 1)  # Assumes 2D, otherwise Nd


# Convenient helper
ChSwap = ChannelSwap()


# Converts data to torch Tensors and moves them to the specified device as floats
def to_torch(xs, device=None):
    return tuple(None if x is None
                 else torch.as_tensor(x, dtype=torch.float32, device=device) for x in xs)


# Converts lists or scalars to tuple, preserving NoneType
def to_tuple(items: (int, float, bool, list, tuple)):
    return None if items is None else (items,) if isinstance(items, (int, float, bool)) else tuple(items)


# Multiples list items or returns item
def prod(items: (int, float, bool, list, tuple)):
    return items if isinstance(items, (int, float, bool)) or items is None else math.prod(items)


# Shifts to positive, normalizes to [0, 1]
class Norm(nn.Module):
    def __init__(self, start_dim=-1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        y = x.flatten(self.start_dim)
        y = y - y.min(-1, keepdim=True)[0]
        y = y / y.max(-1, keepdim=True)[0]
        return y.view(*x.shape)


# Pytorch incorrect (in this case) warning suppression
warnings.filterwarnings("ignore", message='.* skipping the first value of the learning rate schedule')


# Scales gradients for automatic mixed precision training speedup, or updates gradients normally
class MixedPrecision:
    def __init__(self):
        self.mixed_precision_enabled = False  # Corresponds to mixed_precision=true
        self.ready = False
        self.models = set()

    @cached_property
    def scaler(self):
        return torch.cuda.amp.GradScaler()  # Gradient scaler to magnify imprecise Float16 gradients

    def enable(self, args):
        self.mixed_precision_enabled = args.mixed_precision and 'cuda' in args.device

    # Backward pass
    def backward(self, loss, retain_graph=False):
        if self.ready:
            loss = self.scaler.scale(loss)
        loss.backward(retain_graph=retain_graph)  # Backward

    # Optimize
    def step(self, model):
        if self.mixed_precision_enabled:
            if self.ready:
                # Model must AutoCast-initialize before first call to update
                assert id(model) in self.models, 'A new model or block is being optimized after the initial learning ' \
                                                 'update while "mixed_precision=true". ' \
                                                 'Not supported by lazy-AutoCast. Try "mixed_precision=false".'
                try:
                    return self.scaler.step(model.optim)  # Optimize
                except RuntimeError as e:
                    if 'step() has already been called since the last update().' in str(e):
                        e = RuntimeError(
                            f'The {type(model)} optimizer is being stepped twice while "mixed_precision=true" is '
                            'enabled. Currently, Pytorch automatic mixed precision only supports stepping an optimizer '
                            'once per update. Try running with "mixed_precision=false".')
                    raise e

            # Lazy-initialize AutoCast context

            forward = model.forward

            # Enable Pytorch AutoCast context
            model.forward = torch.autocast('cuda', dtype=torch.float16)(forward)

            for module in model.children():  # In case parts are shared across blocks e.g. Discrete Critic <- Actor
                forward = module.forward

                module.forward = torch.autocast('cuda', dtype=torch.float16)(forward)

            # EMA
            if hasattr(model, 'ema'):
                forward = model.ema.forward

                model.ema.forward = torch.autocast('cuda', dtype=torch.float16)(forward)

            self.models.add(id(model))

        model.optim.step()  # Optimize

    def update(self):
        if self.ready:
            self.scaler.update()  # Update gradient scaler
        self.ready = True


MP = MixedPrecision()  # AutoCast + GradScaler automatic mixed precision training speedup via gradient scaling


# Backward pass on a loss; clear the grads of models; update EMAs; step optimizers and schedulers
def optimize(loss, *models, clear_grads=True, backward=True, retain_graph=False, step_optim=True, epoch=0, ema=True):
    # Clear grads
    if clear_grads and loss is not None:
        for model in models:
            if model.optim:
                model.optim.zero_grad(set_to_none=True)

    # Backward
    if backward and loss is not None:
        MP.backward(loss, retain_graph)  # Backward pass

    # Optimize
    if step_optim:
        for model in models:
            # Step scheduler
            if model.scheduler and epoch > model.scheduler.last_epoch:
                model.scheduler.step()
                model.scheduler.last_epoch = epoch

            # Update EMA target
            if ema and hasattr(model, 'ema'):
                update_ema_target(source=model, target=model.ema, ema_decay=model.ema_decay)

            if model.optim:
                MP.step(model)  # Step optimizer

                if loss is None and clear_grads:
                    model.optim.zero_grad(set_to_none=True)


# Increment/decrement a value in proportion to a step count and a string-formatted schedule
def schedule(schedule, step):
    try:
        return float(schedule)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schedule)
        if match:
            start, stop, duration = [float(g) for g in match.groups()]
            mix = float(np.clip(step / duration, 0.0, 1.0))
            return (1.0 - mix) * start + mix * stop
