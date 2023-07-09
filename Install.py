import glob
import os

os.system(f'git pull')

builds = glob.glob('*/dist/*')

with open('install.sh', 'w') as f:
    f.write(f"""pip install {' '.join(builds)} --force-reinstall --no-dependencies""")

os.system('sh install.sh')
