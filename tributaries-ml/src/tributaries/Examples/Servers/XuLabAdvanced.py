# Tutorial 4. Sbatch scripts and plotting as decorator argument

from tributaries import my_server


@my_server('../Sweeps/Bittle', plot=True, checkpoints=False, node='sclera')
def main(node):
    server, username, password = 'slurm', 'slerman', ''
    app_name_paths = {'XRDs': f"/home/cxu-serve/u1/{username}/XRDs/XRD.py"}  # Defines the name and location of apps
    conda = 'conda activate Sam'

    nodes_partitions = {'macula': 'macula', 'cornea': 'gpu', 'iris': 'gpu', 'retina': 'gpu',
                        'sclera': 'macula', 'cxu4090-1': 'macula', 'cxu4090-2': 'macula'}
    sbatch = f'#SBATCH --nodelist {node} --partition {nodes_partitions[node]}'

    # hyper = 'path=/localdisk3 log_path=/u/slerman/u1/Builder/'  # Run in /localdisk. Log in /u1
    hyper = 'path=/u/slerman/u1/Builder/'  # Run and log in /u1

    return server, username, password, None, app_name_paths, conda, sbatch, hyper


if __name__ == '__main__':
    main()

# Command-line examples:

#   Run a sweep like this:
#       $ python XuLabAdvanced.py
#   Plot it:
#       $ python XuLabAdvanced.py plot=true
