import glob
import os

os.system(f'git pull')

builds = glob.glob('*/dist/*')

os.system(f'pip install {" ".join(builds)} --force-reinstall --no-dependencies')
