import glob
import os

os.system(f'git pull')
builds = glob.glob('*/Build.py')

for build in builds:
    os.system(f'python {build}')
