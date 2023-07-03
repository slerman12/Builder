from tributaries import my_server


@my_server('../Sweeps/XRDsPaper')
def main():
    server, username, password = 'slurm', 'slerman', ''
    app_name_paths = {'XRDs': f"/home/cxu-serve/u1/{username}/XRDs/XRD.py"}  # Defines the name and location of apps
    conda = 'conda activate Sam'

    return server, username, password, None, app_name_paths, conda, None


if __name__ == '__main__':
    main()
