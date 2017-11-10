import numpy as np

from . import _ckmeans_wrapper
from ._utils import KmeansResult

def ckmeans(x, k=(1,10), weights=None):
    """Wrapper around the Ckmeans.1d.dp algorithm written in C++

    See https://cran.r-project.org/web/packages/Ckmeans.1d.dp

    Parameters
    ----------
    x : np.array
        1-D array of sortable data
    k : (int, int) or int
        Min k, max k
    weights : np.array
        1-D array of weights (defaults to all equal weights if None)

    Returns
    -------
    KmeansResult
    """
    x = np.array(x, dtype=np.double, order='C', ndmin=1)

    try:
        min_k, max_k = k
    except:
        min_k, max_k = k, k

    if min_k < 1:
        raise ValueError("k must be positive")
    if max_k > len(set(x)):
        raise ValueError("k cannot be greater than the number of unique data points")

    if weights is None:
        weights = np.array([1.0], dtype=np.double)
    elif not isinstance(weights, np.ndarray):
        raise TypeError('weights needs to be a numpy array')

    return _ckmeans_wrapper.ckmeans(x, min_k, max_k, weights)
