import os
import sys
import setuptools

os.chdir(os.path.dirname(__file__))
sys.argv.append('bdist_wheel')
setuptools.setup()  # Build package

# Install package
os.chdir('./..')
package = os.path.dirname(__file__).rsplit('/')[-1]
install = sorted(os.listdir(os.path.dirname(__file__) + '/dist'))[-1]

if os.path.exists(f'{package}/dist/{install}'):
    os.system(f'pip install {package}/dist/{install} --force-reinstall --no-dependencies')
    print(f'pip installed {package} updates. ✓')
else:
    print(f'Could not find {package}/dist/{install}')
