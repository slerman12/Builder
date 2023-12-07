# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
A simple port/mill operating on the branching rivers of data flow between remote servers. Hand-built by Sam Lerman.
"""

import os
import re
from inspect import signature
from math import inf
import shlex
import subprocess
import sys
from functools import partial

import ast
from pexpect import pxssh, spawn

from ML import __file__, plot as make_plots
from minihydra import just_args, instantiate, interpolate, Args, recursive_update, recursive_Args


def sbatch_deploy(hyperparams, deploy_config):
    sys.argv = sys.argv[:1] + shlex.split(hyperparams) + \
               (deploy_config.hyper.split() if deploy_config.hyper else [])

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
#SBATCH -c {getattr(deploy_config, 'num_cpus', None) or args.num_workers + 1}
{f'#SBATCH -p gpu --gres=gpu:{deploy_config.num_gpus}' if deploy_config.num_gpus else ''}
{f'#SBATCH -p reserved --reservation={deploy_config.username}-{deploy_config.reservation_id}'
    if deploy_config.reservation_id else ''}
#SBATCH -t {deploy_config.time} -o {args.logger.path}{args.task_name}_{args.seed}_Log.txt -J {deploy_config.pseudonym or
                                                                                              args.task_name}
#SBATCH --mem={deploy_config.mem}gb 
{extra}
{deploy_config.sbatch if deploy_config.sbatch else ''}
{commands}
{'wandb login ' + deploy_config.wandb_key if deploy_config.wandb_key else ''}
{'python ' + deploy_config.app_name_paths[deploy_config.app] if deploy_config.app_name_paths and deploy_config.app
    else 'ML'} {' '.join(hyperparams.split())} {deploy_config.hyper or ''}
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
def mass_deploy():  # TODO if server= arg included, run as if deploying from local
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

    sweep = recursive_Args(sweep)

    from tributaries.Sweeps import my_sweep

    # Defaults in case tributaries called directly (without sweep)
    defaults = Args(**my_sweep, **{'app_name_paths': None, 'commands': [], 'sbatch': ''})
    sweep.update({key: value for key, value in defaults.items() if key not in sweep})

    if 'path' in [key_value.split('=')[0] for key_value in (sweep.hyper or '').split()]:
        try:
            chdir = [key_value.split('=')[1]
                     for key_value in (sweep.hyper or '').split() if key_value.split('=')[0] == 'path'][0]
            print(f'Changing directory to {chdir}')
            os.chdir(chdir)  # Note: will only be able to change dir into visible disks
        except FileNotFoundError:
            pass

    print(f'Deploying {len(sweep.hyperparams)} set(s) of hyperparams.')

    for i, hyperparams in enumerate(sweep.pop('hyperparams')):
        print(f'Set: {i + 1},', hyperparams)
        sbatch_deploy(hyperparams, sweep)


def launch_remote(server, username, password, sweep):
    # SSH login
    print('\nConnecting to remote server', end=" ")
    ssh = pxssh.pxssh(timeout=100, maxread=200000)
    ssh.login(server, username, password)
    ssh.setwinsize(60000, 60000)  # Allow longer-length commands
    ssh.prompt()
    print('- Connected! ✓\n')
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
    # Mass-deploy via tributaries  TODO SFTP the command in a file
    cmd = ' '.join([f'{key}={int.from_bytes(str(value).encode("utf-8"), "little")}'
                    for key, value in sweep.items()])  # Encode sweep for ssh command-line
    print('Sending command...')
    print('If this takes too long, you may run it manually on your remote server with:', 'tributaries ' + cmd)
    ssh.sendline('tributaries ' + cmd)
    ssh.expect(['tributaries ' + cmd], timeout=500)
    ssh.prompt()
    prompt = ssh.before.decode("utf-8")
    if 'Deploying' not in str(prompt):
        print('Original error:', str(prompt), '\n')
        raise EnvironmentError('Could not launch tributaries on remote server. ' \
                               'Make sure you have installed tributaries ' \
                               '(pip install tributaries) on your remote server and/or ' \
                               'included a commands flag for activating a tributaries-installed ' \
                               f'Python/Conda environment in your remote config.')
    print(prompt)


def download(server, username, password, sweep, plots=None, checkpoints=None):
    original_path = sweep.app_name_paths.get(sweep.app, '')

    if sweep.hyper is not None and 'log_path' in [arg.split('=')[0] for arg in sweep.hyper.split()]:
        sweep.app_name_paths['path'] = [arg.split('=')[1]
                                        for arg in sweep.hyper.split() if arg.split('=')[0] == 'log_path'][0]
        sweep.app = 'path'
    elif sweep.hyper is not None and 'path' in [arg.split('=')[0] for arg in sweep.hyper.split()]:
        sweep.app_name_paths['path'] = [arg.split('=')[1]
                                        for arg in sweep.hyper.split() if arg.split('=')[0] == 'path'][0]
        sweep.app = 'path'

    plots = [] if plots is None else plots.plots
    checkpoints = [] if checkpoints is None else checkpoints.experiments

    experiments = set().union(*plots, checkpoints)

    cwd = os.getcwd()

    # SFTP
    print(f'SFTP\'ing: {", ".join(experiments)}')
    print('\nConnecting to remote server', end=" ")
    p = spawn(f'sftp {username}@{server}')
    if password:
        p.expect('Password: ', timeout=None)
        p.sendline(password)
        p.expect('sftp> ', timeout=None)
    print('- Connected! ✓\n')
    path = sweep.app_name_paths.get(sweep.app, '')
    p.sendline(f'cd {os.path.dirname(path) if ".py" in path else path}')
    p.expect('sftp> ', timeout=None)
    if plots:
        os.makedirs('./Benchmarking/Logs', exist_ok=True)
        os.chdir('./Benchmarking/Logs')
        p.sendline(f'lcd {cwd}/Benchmarking/Logs')
        p.expect('sftp> ', timeout=None)
        for i, experiment in enumerate(experiments):
            print(f'{i + 1}/{len(experiments)} SFTP\'ing "{experiment}"')
            p.sendline(f'get -r ./Benchmarking/Logs/{experiment.replace(".*", "*")}')  # Some regex compatibility
            p.expect('sftp> ', timeout=None)
    p.sendline(f'ls')  # Re-sync
    p.expect('sftp> ', timeout=None)
    if checkpoints:
        os.makedirs(f'{cwd}/Checkpoints', exist_ok=True)
        os.chdir(f'{cwd}/Checkpoints')
        p.sendline(f'lcd {cwd}/Checkpoints')
        p.expect('sftp> ', timeout=None)
        p.sendline(f'cd ~/')
        p.expect('sftp> ', timeout=None)
        p.sendline(f'cd {os.path.dirname(original_path) if ".py" in original_path else original_path}')
        p.expect('sftp> ', timeout=None)
        for i, experiment in enumerate(experiments):
            print(f'{i + 1}/{len(experiments)} SFTP\'ing "{experiment}"')
            p.sendline(f'get -r ./Checkpoints/{experiment.replace(".*", "*")}')  # Some regex compatibility
            p.expect('sftp> ', timeout=None)
    p.sendline(f'ls')  # Re-sync
    p.expect('sftp> ', timeout=None)
    print()
    os.chdir(cwd)


def paint(plots, name=''):
    for plot_train in [False, True]:
        print(f'\n Plotting {"train" if plot_train else "eval"}...')

        for plot_experiments in plots.plots:

            make_plots(path=f"./Benchmarking/{name}/Plots/{'_'.join(plot_experiments).strip('.')}/",
                       plot_experiments=plot_experiments if len(plot_experiments) else None,
                       plot_agents=plots.agents if 'agents' in plots and len(plots.agents) else None,
                       plot_suites=plots.suites if 'suites' in plots and len(plots.suites) else None,
                       plot_tasks=plots.tasks if 'tasks' in plots and len(plots.tasks) else None,
                       steps=plots.steps if 'steps' in plots and plots.steps else inf,
                       write_tabular=getattr(plots, 'write_tabular', False), plot_train=plot_train,
                       title=getattr(plots, 'title', 'UnifiedML'), x_axis=getattr(plots, 'x_axis', 'Step'),
                       verbose=True
                       )


def decorate(server, sweep=None, plot=False, checkpoints=False, **kwargs):
    args = just_args()

    if 'sweep' in args:
        sweep = args.sweep

    if 'plot' in args:
        plot = args.plot

    if 'checkpoints' in args:
        checkpoints = args.checkpoints

    if (plot or checkpoints) and sweep is None:
        sweep = ''
    elif sweep is None:
        assert False, 'A sweep= path must be provided as argument to the server decorator or via command-line.'

    github = getattr(args, 'github', True)
    sftp = getattr(args, 'sftp', True)

    args = {key: value for key, value in args.items() if key not in ['sweep', 'plot', 'checkpoints', 'github', 'level']}

    path = sweep
    sweep = instantiate(sweep + '.my_sweep') if sweep else Args(**args)
    args = {key: args[key] for key in args.keys() & signature(server).parameters}
    config = server(**args, **kwargs)

    config += (None,) * (8 - len(config))
    server, username, password, func, app_name_paths, commands, sbatch, hyper = config

    # TODO No need for app_name_paths; just path or allow task= to infer path. Default path for World/benchmarking
    #  can be in root dir under Tributaries/

    recursive_update(sweep, {'app_name_paths': app_name_paths, 'commands': commands, 'sbatch': sbatch, 'hyper': hyper,
                             'github': github, 'username': username})

    # Call func first
    if func is not None:
        func()

    if plot or checkpoints:
        plots = Args(plots=plot if isinstance(plot[0], (list, tuple)) else [plot]) if isinstance(plot, list) \
            else instantiate(path + '.my_plots') if plot else None
        checkpoints = Args(experiments=checkpoints) if isinstance(checkpoints, list) \
            else instantiate(path + '.my_checkpoints') if checkpoints else None
        if sftp:
            download(server, username, password, sweep, plots, checkpoints)
        if plot:
            name = '/'.join(path.replace('.py', '').replace('.', '/').rsplit('/', sweep.level)[1:]) if path \
                else 'Downloaded'
            paint(plots, name)
    else:
        launch_remote(server, username, password, sweep)


# Decorator for defining servers
def my_server(sweep=None, plot=False, checkpoints=False, **kwargs):
    def decorator_func(server):
        return partial(decorate, server, sweep, plot, checkpoints, **kwargs)
    return decorator_func
