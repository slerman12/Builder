# Tutorial 3. Adding a default sweep, VPN login, custom command-line args, and custom sbatch code

from tributaries.Central import my_server
from tributaries.SafePass import get_pass
from tributaries.VPN import connect_vpn


@my_server('../Sweeps/XRDsPaperBluehive', plot=False, group_name='acmml', username='slerman', env=None)
def main(group_name, username, env):  # Accepts custom args
    server, password = 'bluehive.circ.rochester.edu', get_pass('bluehive')

    vpn = connect_vpn(username)  # Returns a func that connects to VPN when called
    app_name_paths = {'XRDs': f"/scratch/{username}/XRDs/XRD.py"}  # Defines the name and location of apps

    # First create a Conda env on Bluehive with tributaries:

    #   module load tmux
    #   tmux new -s Install
    #   module load miniforge3/22.11.1-2
    #   conda create -n ML python=3.11.3 pip
    #   conda activate ML
    #   pip install tributaries --no-cache-dir --user
    #   pip install tributaries --force-reinstall --no-dependencies
    #   pip install UnifiedML --force-reinstall --no-dependencies

    # If you need help logging into Bluehive, run 'python VPN.py'cop-

    commands = ['module load git',
                'module load miniforge3/22.11.1-2',
                f'conda activate {env or "ML"}']  # Different conda env depending on env= arg

    sbatch = '#SBATCH -p csxu -A cxu22_lab\nmodule load gcc' if group_name == 'csxu' \
        else '#SBATCH -p acmml2 -A acmml2\nmodule load gcc' if group_name == 'acmml' \
        else ''  # Different config depending on group name

    return server, username, password, vpn, app_name_paths, commands, sbatch


if __name__ == '__main__':
    main()

# Command-line examples:

#   Run the default XRDsPaper sweep like this:
#       $ python Bluehive.py
#   Plot it:
#       $ python Bluehive.py plot=true
