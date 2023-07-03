from tributaries.Central import my_server
from tributaries.SafePass import get_pass
from tributaries.VPN import connect_vpn


@my_server('../Sweeps/XRDsPaper')
def main(group_name=None, username='slerman', env=None):  # Can pass in special args by command line
    server, password = 'bluehive.circ.rochester.edu', get_pass('bluehive')

    vpn = connect_vpn(username)  # Returns a func that connects to VPN when called
    app_name_paths = {'XRDs': f"/scratch/{username}/XRDs/XRD.py"}  # Defines the name and location of apps

    conda = f'conda activate {env or "ML"}'  # Different conda env depending on env= arg

    sbatch = ''.join([f'*"{gpu}"*)\n{conda} {env}\n;;\n'
                      for gpu, env in [('K80', env or 'CUDA10.2'),  # Conda envs w.r.t. GPU
                                       ('', env or 'ML')]])  # Default: ML
    sbatch = f'GPU_TYPE' \
             f'=$(nvidia-smi --query-gpu=gpu_name --format=csv | tail  -1)\ncase $GPU_TYPE in\n{sbatch}esac\n'
    sbatch += '#SBATCH -p csxu -A cxu22_lab\nmodule load gcc' if group_name == 'csxu' \
        else '#SBATCH -p acmml -A acmml2\nmodule load gcc' if group_name == 'acmml' \
        else ''  # Different config depending on group name

    return server, username, password, vpn, app_name_paths, conda, sbatch


if __name__ == '__main__':
    main()
