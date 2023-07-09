import glob
import os

os.system(f'git pull')

builds = glob.glob('*/Build.py')

for build in builds:
    os.system(f'python {build}')

builds = glob.glob('*/dist/*')

os.system(f'pip install {" ".join(builds)} --force-reinstall --no-dependencies')
