# Example I followed:
# https://github.com/SeTeM/pync/blob/master/setup.py

import os
from setuptools import setup

file_contents = []
for file_name in ('README.md',):
    path = os.path.join(os.path.dirname(__file__), file_name)
    file_contents.append(open(path).read())
long_description = '\n\n'.join(file_contents)

setup(
    name='ckmeans',
    version='0.1.0',
    description='Python implementation of Ckmeans.1D.DP',
    long_description=long_description,
    author='Greg Werbin',
    author_email='greg@rocketrip.com',
    license='Rocketrip PIIA',
    packages=['ckmeans'],
    install_requires=['numpy'],
    # how to specify R and Ckmeans.1d.dp as well? info at section `test_loader`
    # on http://pythonhosted.org/setuptools/setuptools.html#command-reference
    tests_require=['rpy2'],
    test_suite='tests.ckmeans.ckmeans_test'
)
