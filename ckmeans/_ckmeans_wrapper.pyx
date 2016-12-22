cimport _ckmeans
cimport numpy as np
import numpy as np
import ctypes
from ._utils import KmeansResult

def ckmeans(np.ndarray[np.float64_t] x,
            int min_k, int max_k,
            np.ndarray[np.float64_t] weights):
    x = np.ascontiguousarray(x, dtype=np.dtype('d'))
    y = np.ascontiguousarray(weights, dtype=np.dtype('d'))

    cdef int n_x = len(x)
    cdef int n_weights = len(weights)

    cdef np.ndarray[int, ndim=1] clustering = np.ascontiguousarray(np.empty((n_x,), dtype=ctypes.c_int))

    # pre-allocate for max k, then truncate later
    cdef np.ndarray[np.double_t, ndim=1] centers = np.ascontiguousarray(np.empty((max_k,), dtype=np.dtype('d')))
    cdef np.ndarray[np.double_t, ndim=1] within_ss = np.ascontiguousarray(np.zeros((max_k,), dtype=np.dtype('d')))  # use `zeros` because `empty` was causing weird problems
    cdef np.ndarray[int, ndim=1] sizes = np.ascontiguousarray(np.zeros((max_k,), dtype=ctypes.c_int))

    cdef double total_ss = 0
    cdef double between_ss = 0

    _ckmeans.Ckmeans_1d_dp(&x[0], &n_x,
                           &weights[0], &n_weights,
                           &min_k, &max_k,
                           &clustering[0], &centers[0],
                           &within_ss[0], &sizes[0])

    print(within_ss)

    if n_x == n_weights and y.sum() != 0:
        total_ss = np.sum(y * (x - np.sum(x * weights) / weights.sum()) ** 2)
    else:
        total_ss = np.sum((x - x.sum() / n_x) ** 2)

    # since we initialized sizes as a vector of 0's, and size can never be
    # zero, we know that any 0 size element is unused
    cdef int k = np.sum(sizes > 0)

    centers = centers[:k]
    within_ss = within_ss[:k]
    sizes = sizes[:k]

    between_ss = total_ss - within_ss.sum()

    # change the clustering back to 0-indexed,
    # since the R wrapper changes it to 1-indexed.
    return KmeansResult(clustering - 1, k, centers, sizes, within_ss, total_ss, between_ss)
