from collections import namedtuple
from unittest import TestCase
from time import perf_counter
from tqdm import tqdm

from rpy2 import robjects
from rpy2.robjects import r as R
from rpy2.robjects.packages import importr

import numpy as np
rand = np.random.RandomState(43770)
from rpy2.robjects import numpy2ri
numpy2ri.activate()
Ckmeans_1d_dp = importr('Ckmeans.1d.dp')

from ckmeans import *

TestData = namedtuple('TestData', ['x', 'k'])
gaussians_2 = np.concatenate((rand.normal(3, 1/2, 15), rand.normal(23, 1/2, 13)))

values = np.array([ 176.2,  205.2,  206.2,  207.2,  226.2,  237.2,  239.2,  249.2,
    252.2,  254.2,  266.2,  269.2,  275.2,  276.2,  279.2,  296.2, 297.2,
    305.2,  306.2,  310.2])
counts = np.array([  6,  14,  84,  20,  84, 280,   4,   2,  20,   4,   8,  12,
    42, 4,  52,  26,  30,  54,  12,  10])
messy = np.concatenate([np.repeat(value, count) for value, count in zip(values, counts)])

test_cases = {
    'gaussians_2': TestData(gaussians_2, 2),
    'guess_k': TestData(messy, (1, 10)),
    'given_k': TestData(np.array([-1, 2, -1, 2, 4, 5, 6, -1, 2, -1]), 3),  # test copied from Ckmeans.1d.dp
}


class TestFramework():
    def compare_arrays(self, my_result, reference_result):
        self.assertSequenceEqual(my_result.shape, reference_result.shape)
        # print('my result:\n{}'.format(my_result))
        # print('reference result:\n{}'.format(reference_result))
        self.assertTrue(np.allclose(my_result, reference_result))


class TestCkmeans(TestCase, TestFramework):
    def compare_results(self, my_result, r_result):
        clustering = np.asarray(r_result.rx2('cluster')) - 1
        centers = np.asarray(r_result.rx2('centers'))
        within_ss = np.asarray(r_result.rx2('withinss'))
        sizes = np.asarray(r_result.rx2('size'))
        self.compare_arrays(my_result.clustering, clustering)
        self.compare_arrays(my_result.centers, centers)
        self.compare_arrays(my_result.within_ss, within_ss)
        self.compare_arrays(my_result.sizes, sizes)

    def run_test(self, test):
        x, k = test_cases[test]
        my_result = do_ckmeans(x, k)
        r_result = Ckmeans_1d_dp.Ckmeans_1d_dp(x, k)
        self.compare_results(my_result, r_result)

    def test_calculate__guess_k__small(self):
        x, k_range = test_cases['guess_k']
        my_results = []
        BICs = []
        pf1 = perf_counter()
        for k in tqdm(range(k_range[0], k_range[-1] + 1)):
            res = do_ckmeans(x, k)
            my_results.append(res)
            BIC = kmeans_BIC(x, res)
            BICs.append(BIC)
        pf2 = perf_counter()
        best = np.array(BICs).argmin()
        my_result = my_results[best]
        pf3 = perf_counter()
        r_result = Ckmeans_1d_dp.Ckmeans_1d_dp(x, np.array((k_range[0], k_range[-1])))
        pf4 = perf_counter()
        try:
            self.compare_results(my_result, r_result)
        except AssertionError as e:
            self.compare_results(my_result, r_result)
            raise e
        print("\nTimings:")
        print("\tCkmeans.1d.dp: {:.8f}".format(pf4 - pf3))
        print("\tRocketrip: {:.8f}".format(pf2 - pf1))

    def test_calculate__guess_k__gaussians_2(self):
        x, k_actual = test_cases['gaussians_2']
        k_range = (1, 10)
        my_results = []
        BICs = []
        pf1 = perf_counter()
        for k in tqdm(range(k_range[0], k_range[-1] + 1)):
            res = do_ckmeans(x, k)
            my_results.append(res)
            BIC = kmeans_BIC(x, res)
            BICs.append(BIC)
        pf2 = perf_counter()
        best = np.array(BICs).argmin()
        my_result = my_results[best]
        pf3 = perf_counter()
        r_result = Ckmeans_1d_dp.Ckmeans_1d_dp(x, np.array((k_range[0], k_range[-1])))
        pf4 = perf_counter()
        self.compare_results(my_result, r_result)
        print("\nTimings:")
        print("\tCkmeans.1d.dp: {:.8f}".format(pf4 - pf3))
        print("\tRocketrip: {:.8f}".format(pf2 - pf1))
        print("Number of clusters:")
        print("\tActual: {}".format(k_actual))
        print("\tCkmeans.1d.dp: {}".format(best + 1))
        print("\tRocketrip: {}".format(len(r_result.rx2('size'))))


