import glob
import os

os.system(f'git pull')

builds = glob.glob('*/Build.py')

for build in builds:
    os.system(f'python {build}')

builds = glob.glob('*/dist/*')

if '--from-scratch' in sys.argv:
    os.system(f'pip install {" ".join(builds)} --force-reinstall')
else:
    os.system(f'pip install {" ".join(builds)} --force-reinstall --no-dependencies')
