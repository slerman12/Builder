# Make sure to install twine: $pip install twine

import glob

builds = glob.glob('*/dist/*')

# Username: __token__
# Password: <see token on PyPi>

# Can change token via account settings on PyPi
# https://pypi.org/manage/account/#api-tokens

# Also have to iterate version in setup.py, re-build, and delete previous distribution

for build in builds:
    print(f'python -m twine upload {build}')

print('\nUsername: __token__'
      '\nPassword: <see token on PyPi>')
