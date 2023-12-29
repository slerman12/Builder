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
      '\nPassword: pypi-AgEIcHlwaS5vcmcCJDkzZDAzMTViLTVlYTUtNDhhNC04YmY0LTY2ODM4NjBiMGU5ZAACKlszLCI5MTE0MzAwMi1iMzI0LTQ'
      '4NzYtOTA3Zi0wOGE0MWYzNzM4NTgiXQAABiAVA6NeKfY62XOl9x5aVTtGulT5pJT5L-TxpwPQo5Uyow')
