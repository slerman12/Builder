import sys
import setuptools

print('Make sure to run Copy.py first, and under the correct branch.')

sys.argv.append('bdist_wheel')
setuptools.setup()
