# Tutorial 4. Sbatch scripts and plotting as decorator argument

from tributaries import my_server


@my_server('../Sweeps/XRDsPaper', plot=False, node='macula')
def main(node):
    server, username, password = 'slurm', 'slerman', ''
    app_name_paths = {'XRDs': f"/home/cxu-serve/u1/{username}/XRDs/XRD.py"}  # Defines the name and location of apps
    conda = 'conda activate Sam'

    nodes_partitions = {'macula': 'macula', 'cornea': 'gpu', 'iris': 'gpu', 'retina': 'gpu'}
    sbatch = f'#SBATCH --nodelist {node} --partition {nodes_partitions[node]}'

    return server, username, password, None, app_name_paths, conda, sbatch


if __name__ == '__main__':
    main()

# Command-line examples:

#   Plot it:
#       $ python XuLab.py
