# Example I followed:
# https://github.com/SeTeM/pync/blob/master/setup.py

import os
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

file_contents = []
for file_name in ('README.md',):
    path = os.path.join(os.path.dirname(__file__), file_name)
    file_contents.append(open(path).read())
long_description = '\n\n'.join(file_contents)

wrapper = Extension(
    name='ckmeans._ckmeans_wrapper',
    sources=[
        'ckmeans/_ckmeans_wrapper.pyx',
        'Ckmeans.1d.dp/src/Ckmeans.1d.dp.cpp',
        'Ckmeans.1d.dp/src/select_levels.cpp',
        'Ckmeans.1d.dp/src/weighted_opt_uni_kmeans.cpp',
        'Ckmeans.1d.dp/src/weighted_select_levels.cpp'
    ],
    language="c++",
    include_dirs=['Ckmeans.1d.dp/src', np.get_include()],
    extra_compile_args=['-std=c++11']
)

setup(
    name='ckmeans',
    version='1.1.1',
    description='Python wrapper around Ckmeans.1d.dp',
    long_description=long_description,
    author='Greg Werbin',
    author_email='greg@rocketrip.com',
    license='MIT',
    packages=['ckmeans'],
    ext_modules = cythonize(wrapper),
    install_requires=['numpy', 'Cython'],
    # how to specify R and Ckmeans.1d.dp as well? info at section `test_loader`
    # on http://pythonhosted.org/setuptools/setuptools.html#command-reference
    tests_require=['rpy2'],
    test_suite='tests.ckmeans.ckmeans_test'
)
