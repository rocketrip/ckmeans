import numpy as np

def BIC(log_likelihood, n_params, n_obs):
    """Schwartz' Bayesian Information Criterion for an arbitrary log-likelihood

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
    """Get BIC from a k-means fit

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


def _choose_k_PDN(x, k_range):
    """Method of Pham, Dimov, and Nguyen for choosing k in k-means

    See https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/

    Parameters
    ----------
    x : np.array
    k_options : collection of k values to try
    """
    raise NotImplementedError()


def _choose_k_gap(x, k_range):
    """Choose k using the "gap statistic" of Tibshirani, Walther, and Hastie

    See http://www.web.stanford.edu/~hastie/Papers/gap.pdf

    Parameters
    ----------
    x : np.array
    k_options : collection of k values to try
    """
    raise NotImplementedError()


def _choose_k_BIC(x, k_options):
    """choose k using bic

    parameters
    ----------
    x : np.array
    k_options : collection of k values to try
    """
    raise NotImplementedError()


def choose_k(k, k_options, method='BIC', *args, **kwargs):
    func_mapping={
        'PDN': _choose_k_pdn,
        'gap': _choose_k_gap,
        'BIC': _choose_k_BIC
    }
    try:
        choose_k_func = func_mapping[method]
    except KeyError:
        raise ValueError("Invalid method {}".format(method))
    return choose_k_func(k, k_options, *args, **kwargs)
