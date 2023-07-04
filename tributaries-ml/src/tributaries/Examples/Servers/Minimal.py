# Tutorial: Bare minimum example

from tributaries import my_server


@my_server()
def main():
    server, username, password = 'slurm', 'slerman', ''

    return server, username, password


if __name__ == '__main__':
    main()


# This will operate in the root directory of the remote server.

# Plot a sweep like this:
#   $ python Minimal.py sweep=../Sweeps/XRDsPaper plot=true
# Run a sweep like this:
#   $ python Minimal.py sweep=../Sweeps/XRDsPaper app_name_paths.XRDs=u1/XRDs/XRD.py app=XRDs
# You can change-dir elsewhere via command-line like this:
#   $ python Minimal.py plot='["Exp"]' app_name_paths.name=u1/Builder app=name
