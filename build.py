import glob
import os, sys

os.system(f'git pull')

builds = glob.glob('*/Build.py')

for build in builds:
    os.system(f'python {build}')

# builds = glob.glob('*/build/')
#
# for build in builds:
#     os.system(f'rm -rf {build}')

builds = glob.glob('*/dist/*')

if '--from-scratch' in sys.argv:
    os.system(f'pip install {" ".join(builds)} --force-reinstall')
else:
    os.system(f'pip install {" ".join(builds)} --force-reinstall --no-dependencies')

# If issues, may be necessary to delete auto-generated build/ directories and re-run.
