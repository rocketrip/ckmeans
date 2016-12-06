Ckmeans.1d.dp is a dynamic programming algorithm for globally-optimal clustering
of one-dimensional vectors. Its reference implementation is in C++, wrapped in
an [R package](https://cran.r-project.org/web/packages/Ckmeans.1d.dp). This
module is a Cython wrapper around the C++ code inside that package.

`pip install 'git+https://github.com/rocketrip/ckmeans/'`

NOTE: this will fail to build using the default compiler (Clang) on OS X El
Capitan. I have not tested it with newer versions of Clang. It will build
successfully with GCC 6 as installed by Homebrew. If you have build problems,
try specifying a different compiler with the `CC` environment variable.

The Ckmeans.1d.dp source tree is version 3.4.6-4, as downloaded from the [CRAN 
archive](https://cran.r-project.org/src/contrib/Archive/Ckmeans.1d.dp/Ckmeans.1d.dp_3.4.6-4.tar.gz).
