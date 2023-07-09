import os

os.system(f'git pull')

# builds = glob.glob('*/Build.py')
#
# for build in builds:
#     os.system(f'python {build}')

with open('install.sh', 'w') as f:
    f.write("""git pull
pip install UnifiedML/dist/UnifiedML-1.0.0-py3-none-any.whl tributaries-ml/dist/tributaries_ml-1.0.0-py3-none-any.whl minihydra/dist/minihydra-1.0.0-py3-none-any.whl --force-reinstall --no-dependencies""")

os.system('sh install.sh')
