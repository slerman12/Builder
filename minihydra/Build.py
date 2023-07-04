import os
import sys
import setuptools

os.chdir(__file__.rsplit('/', 1)[0])
sys.argv.append('bdist_wheel')
setuptools.setup()
