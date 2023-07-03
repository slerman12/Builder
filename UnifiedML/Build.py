import os
import sys
import setuptools

print('Make sure to run Copy.py first, and under the correct branch.')

os.chdir(__file__.rsplit('/', 1)[0])
sys.argv.append('bdist_wheel')
setuptools.setup()
