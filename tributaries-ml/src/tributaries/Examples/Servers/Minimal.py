# Tutorial 1. Bare minimum example

from tributaries import my_server


@my_server()
def main():
    server, username, password = 'my_domain_example.com', 'slerman', ''

    return server, username, password


if __name__ == '__main__':
    main()

# Since no app_name_paths provided, this will default to the root directory of the remote server.

# Command-line examples:

#   Plot a sweep like this:
#       $ python Minimal.py sweep=../Sweeps/XRDsPaper plot=true
#   Run a sweep like this:
#       $ python Minimal.py sweep=../Sweeps/XRDsPaper app_name_paths.XRDs=u1/XRDs/XRD.py app=XRDs
#   You can change-dir elsewhere via command-line like this:
#       $ python Minimal.py plot='["Exp"]' app_name_paths.name=u1/Builder app=name
