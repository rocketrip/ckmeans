from collections import namedtuple
from warnings import warn

import numpy as np

"""Cluster result container
"""
ClusterResult = namedtuple('ClusterResult', ['clustering', 'centers', 'within_ss', 'sizes'])


def dynamic_optimize_python(x, k):
    n = len(x)
    D = np.zeros((n, k), dtype=np.float64)
    B = np.zeros((n, k), dtype=np.intp)
    # D[i, m] = min_{m <= j <= i} D[j - 1, m - 1] + d(x_j, ... x_i)
    #   for 1 <= i <= n, 1 <= m <= k
    # where d(x_j, ..., x_i) = sum of sq. distances of x's from their mean
    for m in range(k):

        if m < k - 1:
            # start at m-1 and not 0, because the DP optimization only looks at
            # j from m to i
            istart = max(1, m)
        else:
            # From the comments in the C++ version:
            #   "No need to compute D[K-1][0] ... D[K-1][N-2]"
            # In this case the inner loop will only iterate once, over the
            # entire data set
            istart = n - 1
        for i in range(istart, n):
            # if m == 0, this is the first cluster and we can take some
            # shortcuts because the range "j to i" is in fact all of x
            if m == 0:
                if i == istart:
                    candidate_mean = x[0]
                d = i / (i + 1) * (x[i] - candidate_mean) ** 2
                D[i, 0] = D[i - 1, 0] + d
                B[i, 0] = 0
                candidate_mean = (i * candidate_mean + x[i]) / (i + 1)
            else:
                # initialize d(x_j, ..., x_i), the "sum of squares from the mean"
                # for this j-to-i group
                d = 0
                # initialize mean(x_j, ..., x_i) for this j-to-i group
                candidate_mean = 0
                for j in range(i, m - 1, -1):
                    # use previously computed values of d and mean_j_i instead of
                    # computing from scratch every time. this reduces runtime from
                    # O(k * n^3) to O(k * n^2)
                    d += (i - j) / (i - j + 1) * (x[j] - candidate_mean) ** 2
                    # update the mean for this cluster candidate
                    candidate_mean = (x[j] + (i - j) * candidate_mean) / (i - j + 1)
                    # j == i means this is the first j we're trying, so just fill
                    # in the matrix and move on (no need for comparisons)
                    if d > D[i, m]:
                        # from the comments in the C++ version:
                        #   "Stop reducing j if putting xj to xi into cluster k
                        #   generates worse d than the best D[k][i] so far
                        break
                    candidate_objective = D[j - 1, m - 1] + d
                    if j == i:
                        D[i, m] = candidate_objective
                        B[i, m] = j
                    else:
                        if candidate_objective <= D[i, m]:
                            D[i, m] = candidate_objective
                            B[i, m] = j
    return D, B


try:
    from numba import jit
    raise ImportError("Testing")
    dynamic_optimize_numba = jit(dynamic_optimize_python, nopython=True)
    dynamic_optimize = dynamic_optimize_numba
except ImportError:
    warn("Numba JIT not available, using (slow) pure-Python version")
    dynamic_optimize = dynamic_optimize_python


def backtrack(x, B):
    k = B.shape[1]
    n = B.shape[0]
    clustering = np.zeros(n)
    sizes = np.zeros(k, dtype=np.int)
    centers = np.zeros(k)
    within_ss = np.zeros(k)
    right = n - 1
    for m in range(k - 1, -1, -1):
        left = B[right, m]
        i_m = slice(left, right + 1)
        x_m = x[i_m]
        clustering[i_m] = m
        size = right - left + 1
        center = x_m.sum() / float(size)
        ss = np.sum((x_m - center) ** 2)
        sizes[m] = size
        centers[m] = center
        within_ss[m] = ss
        if m > 0:
            right = left - 1
    return clustering, centers, within_ss, sizes


def recover_order(x, x_sorted, k, sizes):
    clustering = np.repeat(-1, len(x))
    for i in range(len(x)):
        left = 0
        for m in range(k):
            right = left + sizes[m] - 1
            if x[i] <= x_sorted[right]:
                clustering[i] = m
                break
            left = right + 1
    return clustering

#
# """
# Hard-coded versions for testing Numba vs CPython
# """
# def do_ckmeans_python(x, k):
#     order = x.argsort()
#     x_sorted = x[order]
#     if k < 1:
#         raise ValueError("k must be positive")
#     if k > len(set(x)):
#         raise ValueError("k cannot be greater than the number of unique data points")
#     D, B = dynamic_optimize_python(x_sorted, k)
#     clustering_sorted, centers, within_ss, sizes = backtrack(x_sorted, B)
#     clustering = recover_order(x, x_sorted, k, sizes)
#     return ClusterResult(clustering, centers, within_ss, sizes)
#
# def do_ckmeans_numba(x, k):
#     order = x.argsort()
#     x_sorted = x[order]
#     if k < 1:
#         raise ValueError("k must be positive")
#     if k > len(set(x)):
#         raise ValueError("k cannot be greater than the number of unique data points")
#     D, B = dynamic_optimize_numba(x_sorted, k)
#     clustering_sorted, centers, within_ss, sizes = backtrack(x_sorted, B)
#     clustering = recover_order(x, x_sorted, k, sizes)
#     return ClusterResult(clustering, centers, within_ss, sizes)
#


def do_ckmeans(x, k):
    """Pure Python implementation of https://cran.r-project.org/web/packages/Ckmeans.1d.dp

    Parameters
    ----------
    x : np.array
        1-D array of sortable data
    k : int
        Number of clusters to use

    Returns
    -------
    ClusterResult
        clustering
            A 1-D Numpy array the same length as x, indicating cluster membership (starting at 0)
        centers
            A 1-D Numpy array with length k, containing cluster centers
        within_ss
            A 1-D Numpy array with length k, containing within-cluster sums of squares
        sizes
            A 1-D Numpy array with length k, containing the number of points in each cluster
    """
    order = x.argsort()
    x_sorted = x[order]
    if k < 1:
        raise ValueError("k must be positive")
    if k > len(set(x)):
        raise ValueError("k cannot be greater than the number of unique data points")
    D, B = dynamic_optimize(x_sorted, k)
    clustering_sorted, centers, within_ss, sizes = backtrack(x_sorted, B)
    clustering = recover_order(x, x_sorted, k, sizes)
    return ClusterResult(clustering, centers, within_ss, sizes)


def BIC(log_likelihood, n_params, n_obs):
    """Schwartz' Bayesian Information Criterion
    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model
    n_params : int
        Number of model parameters
    n_obs : int
        Number of observations
    """
    return -2 * log_likelihood + n_params * np.log(n_obs)


def kmeans_BIC(x, cluster_result):
    """Choose K for K-means using Bayesian Information Criterion
    Parameters
    ----------
    x : np.array
        1-D array of sortable data
    cluster_result : ClusterResult
        Cluster results fitted on x. Note that no checking is done to ensure
        that the two arguments actually correspond!
    """
    k = len(cluster_result.sizes)
    n = len(x)
    ll = 0
    i_left = 0
    for m in range(k):
        size = cluster_result.sizes[m]
        within_ss = cluster_result.within_ss[m]
        variance = within_ss / (size - 1) if size > 1 else 0
        order = x.argsort()
        x_sorted = x[order]
        n_unique = len(set(x[cluster_result.clustering == m]))
        i_right = i_left + size - 1
        if np.allclose(variance, 0):
            if n_unique == 1:
                x_left = np.mean((x_sorted[i_left-1], x_sorted[i_left])) if i_left > 0 else x[0]
                x_right = np.mean((x_sorted[i_right], x_sorted[i_right+1])) if i_right < n-1 else x[-1]
            else:
                x_left = x_sorted[i_left]
                x_right = x_sorted[i_right]
            x_width = x_right - x_left
            ll += size * np.log(1 / x_width / size)
        else:
            ll += - within_ss.sum() / (2 * variance)
            ll += size * (np.log(size / n) - 0.5 * np.log (2 * np.pi * variance))
        if np.isnan(ll):
            raise ValueError('infinite LL')
        i_left = i_right + 1
    bic = BIC(ll, 3 * k - 1, n)
    return bic


def kmeans_plusplus(x, cluster_result):
    # https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/
    raise NotImplementedError()


