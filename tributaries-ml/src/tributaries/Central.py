# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
A lightweight tool for mass-deploying and plotting ML experiments on slurm-enabled servers. Programmed by Sam Lerman.
"""

import os
import re
import shlex
import subprocess
import sys
from functools import partial

import ast
from pexpect import pxssh

from ML import __file__, import_paths
from ML.Utils import grammars
from ML.Hyperparams.minihydra import just_args, instantiate, interpolate, yaml_search_paths, grammar


def sbatch_deploy(hyperparams, deploy_config):
    sys.argv = sys.argv[:1] + shlex.split(hyperparams)

    args = just_args(os.path.dirname(__file__) + '/Hyperparams/args.yaml')

    os.makedirs(args.logger.path, exist_ok=True)

    # Allow naming tasks with minihydra interpolation syntax
    if args.logger.path and (re.compile(r'.+\$\{[^((\$\{)|\})]+\}.*').match(args.logger.path) or
                             re.compile(r'.*\$\{[^((\$\{)|\})]+\}.+').match(args.logger.path)):
        deploy_config.pseudonym = interpolate([args.logger.path], args)[0]

    # Allow naming tasks with minihydra interpolation syntax
    if deploy_config.pseudonym and (re.compile(r'.+\$\{[^((\$\{)|\})]+\}.*').match(deploy_config.pseudonym) or
                                    re.compile(r'.*\$\{[^((\$\{)|\})]+\}.+').match(deploy_config.pseudonym)):
        deploy_config.pseudonym = interpolate([deploy_config.pseudonym], args)[0]

    # Can add GPU to python script e.g. experiment='name_$GPU_TYPE'

    # Should work in the general case; currently only works on some remote servers and not others
    # extra = f'#SBATCH -C {deploy_config.gpu}'
    extra = ''  # Ignoring deploy_config.gpu for now TODO

    commands = '\n'.join(deploy_config.commands)

    script = f"""#!/bin/bash
#SBATCH -c {args.num_workers}
{f'#SBATCH -p gpu --gres=gpu:{deploy_config.num_gpus}' if deploy_config.num_gpus else ''}
{f'#SBATCH -p reserved --reservation={deploy_config.username}-{deploy_config.reservation_id}'
    if deploy_config.reservation_id else ''}
#SBATCH -t {deploy_config.time} -o {args.logger.path}{args.task_name}_{args.seed}.log -J {deploy_config.pseudonym or
                                                                                          args.task_name}
#SBATCH --mem={deploy_config.mem}gb 
{extra}
{deploy_config.sbatch if deploy_config.sbatch else ''}
{commands}
{'wandb login ' + deploy_config.wandb_key if deploy_config.wandb_key else ''}
{'python ' + deploy_config.app_name_path[deploy_config.app] if deploy_config.app_name_path and deploy_config.app
    else 'ML'} {hyperparams}
"""

    # Write script
    with open('./sbatch_script', 'w') as file:
        file.write(script)

    # Launch script (with error checking / re-launching)
    while True:
        try:
            success = str(subprocess.check_output([f'sbatch ./sbatch_script'], shell=True))
            print(success[2:][:-3])
            if "error" not in success:
                break
        except Exception:
            pass
        print("Errored... trying again")
    print("Success!")


# Works as just sbatch launcher as well, e.g. tributaries hyperparams='...' app=run.py
def mass_deploy():
    import_paths(yaml_search_paths)  # TODO Not sure why this is needed explicitly
    grammars(grammar)  # TODO Is this needed explicitly if Utils is imported?

    sweep = just_args()

    if 'hyperparams' not in sweep:
        sweep.hyperparams = ['']

    if isinstance(sweep.hyperparams, int):
        for key, value in sweep.items():
            try:
                sweep[key] = ast.literal_eval(value.to_bytes((value.bit_length() + 7) // 8, 'little').decode('utf-8'))
            except (ValueError, SyntaxError):
                sweep[key] = str(value.to_bytes((value.bit_length() + 7) // 8, 'little').decode('utf-8'))

    if isinstance(sweep.hyperparams, str):
        sweep.hyperparams = [sweep.hyperparams]

    from tributaries.Sweeps import my_sweep

    # Defaults in case tributaries called directly (without sweep)
    defaults = {**my_sweep, **{'app_name_path': None, 'commands': [], 'sbatch': ''}}
    sweep.update({key: value for key, value in defaults.items() if key not in sweep})

    print(f'Deploying {len(sweep.hyperparams)} set(s) of hyperparams.')

    for i, hyperparams in enumerate(sweep.pop('hyperparams')):
        print(f'Set: {i + 1},', hyperparams)
        sbatch_deploy(hyperparams, sweep)


def launch_remote(server, username, password, sweep):
    # SSH login
    ssh = pxssh.pxssh()
    ssh.login(server, username, password)
    # Go to app
    if sweep.app_name_paths and sweep.app:
        ssh.sendline(f'cd {sweep.app_name_paths[sweep.app].rsplit("/", 1)[0]}')
        ssh.prompt()
        print(ssh.before.decode("utf-8"))
        # Checkout git branch and sync with GitHub first
        if sweep.github:
            # Switch git branch
            if sweep.branch:
                ssh.sendline(f'git fetch origin')
                ssh.prompt()
                print(ssh.before.decode("utf-8"))
                ssh.sendline(f'git checkout -b {sweep.branch} origin/{sweep.branch}')
                ssh.prompt()
                prompt = ssh.before.decode("utf-8")
                if f"fatal: A branch named '{sweep.branch}' already exists." in prompt:
                    ssh.sendline(f'git checkout {sweep.branch}')
                    ssh.prompt()
                    prompt = ssh.before.decode("utf-8")
                print(prompt)
            # GitHub sync
            ssh.sendline(f'git pull origin {sweep.branch}')
            ssh.prompt()
            print(ssh.before.decode("utf-8"))
    if isinstance(sweep.commands, str):
        sweep.commands = [sweep.commands]  # Commands can be string or list
    # Send command-line commands first
    for command in sweep.commands:
        ssh.sendline(command)
        ssh.prompt()
        print(ssh.before.decode("utf-8"))
    # Mass-deploy via tributaries
    cmd = ' '.join([f'{key}={int.from_bytes(str(value).encode("utf-8"), "little")}'
                    for key, value in sweep.items()])  # Encode sweep for ssh command-line
    ssh.sendline('tributaries ' + cmd)
    ssh.prompt()
    prompt = ssh.before.decode("utf-8")
    assert 'Deploying' in str(prompt), 'Could not launch tributaries on remote server. ' \
                                       'Make sure you have installed tributaries ' \
                                       '(pip install tributaries) on your remote server and/or ' \
                                       'included commands for activating a tributaries-installed ' \
                                       f'Python environment in your remote config. Error: {str(prompt)}'
    print(prompt)


def decorate(server, sweep=None, plot=False, checkpoints=False):
    args = just_args()

    if 'sweep' in args:
        sweep = args.sweep

    assert sweep is not None, 'A sweep= path must be provided as argument to the server decorator or via command-line.'

    if 'plot' in args:
        plot = args.plot

    if 'checkpoints' in args:
        checkpoints = args.checkpoints

    # TODO Plotting
    # TODO Checkpoints

    github = getattr(args, 'github', True)

    args = {key: value for key, value in args.items() if key not in
            ['sweep', 'plot', 'plot_sweep', 'checkpoints', 'checkpoints_sweep', 'github']}

    # TODO This kind of dynamic pathfinding should be part of minihydra
    if '/' in sweep:
        root, sweep = sweep.rsplit('/', 1)
        if not os.path.exists(root):
            root = os.getcwd() + '/' + root
        os.chdir(root)  # Makes server-relative paths possible

    sweep = instantiate(sweep + '.my_sweep')
    config = server(**args)

    if len(config) == 5:
        (server, username, password, func, app_name_paths), commands, sbatch = config, None, None
    else:
        server, username, password, func, app_name_paths, commands, sbatch = config

    sweep.update({'app_name_paths': app_name_paths, 'commands': commands, 'sbatch': sbatch,
                  'github': github, 'username': username})

    # Call func first
    if func is not None:
        func()

    launch_remote(server, username, password, sweep)


# Decorator for defining servers
def my_server(sweep=None, plot=False, checkpoints=False):
    def decorator_func(server):
        return partial(decorate, server, sweep, plot, checkpoints)
    return decorator_func
