# Tributaries 

A library for mass-deploying [UnifiedML](github.com/agi-init/UnifiedML) apps on [slurm](https://en.wikipedia.org/wiki/Slurm_Workload_Manager)-enabled remote servers.

```console
pip install tributaries-ml
```

Simply create and run a python file with a server configuration like this one:

```python
# MyServer.py

from tributaries import my_server


@my_server(sweep='path/to/my/sweep.py')
def main():
    ...
    return server, username, password, func, app_name_paths, commands, sbatch


if __name__ == '__main__':
    main()
```

That method must return the ```server```, ```username```, ```password```, any additional ```func``` that needs to be run (e.g. [connecting to a VPN](VPN.py)), and a ```app_name_paths``` dictionary of names and paths to any UnifiedML apps you'd like to use, *e.g.* ```{'name_of_my_app': '/server/path/to/name_of_my_app/run.py'}```.

Optionally:
- A ```commands``` list or string of any extra environment-setup commands you may need to pass to the remote server command-line and deploy config such as [activating a conda environment for example](Examples/Servers/XuLab.py#L9).
- Any additional ```sbatch``` string text you'd like to add to the deploy config.

[You may use one of the blueprint server files provided.](Examples/Servers)

Note the decorator may accept a ```sweep=``` file path for picking out the hyperparams to launch the experiments with.

You can define a ```sweep``` file like this one:

```python
# path/to/my/sweep.py

from tributaries import my_sweep

my_sweep.hyperparams.extend(['...', '...'])  # List of hyperparams
my_sweep.app = 'name_of_my_app'
```

[You may use one of the blueprint sweep files examples](Examples/Sweeps) to make it easy.

You can also pass in the sweep file path via command line with the ```sweep=path.to.my.sweep``` flag. The command-line flag will override the decorator flag; therefore the decorator flag is optional if the command-line flag is present.

That's it. Running it via ```python MyServer.py``` will launch the corresponding sweep experiments on your remote server. Add the ```plot=true``` flag to instead download plots back down to your local machine.

Add ```checkpoints=true``` to download checkpoints.

#### Launching

```console
python MyServer.py sweep=path.to.my.sweep
```

#### Plotting

```console
python MyServer.py sweep=path.to.my.sweep plot=true
```

[//]: # (Note: these hyperparams are already fully part of [UnifiedML]&#40;github.com/agi-init/UnifiedML&#41;, together with the ```my_server=``` server-path flag for pointing to a server file, *e.g.*, ```ML my_server=MyServer.main``` can launch and plot the above directly from [UnifiedML]&#40;github.com/agi-init/UnifiedML&#41;! )

Note: Tributaries launching fully works for non-UnifiedML apps too. Also, for convenience, ```tributaries hyperparams='...' app='run.py'``` can be used as a general slurm launcher on your remote servers.

One more thing: if your remote UnifiedML apps are [git-ssh enabled](https://docs.github.com/en/authentication/connecting-to-github-with-ssh), Tributaries will automatically try syncing with the latest branch via a git pull. You can disable automatic GitHub-syncing with the ```github=false``` flag. 

#

[Licensed under the MIT license.](MIT_LICENSE)

<img width="10%" alt="tributaries-logo" src="https://github.com/AGI-init/Assets/assets/92597756/7e7bb054-f265-4f53-a4f2-d3af52f1d890">