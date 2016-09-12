from collections import namedtuple
from unittest import TestCase

import numpy as np
import numpy.testing
rand = np.random.RandomState(43770)

from rpy2 import robjects
from rpy2.robjects import r as R
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
Ckmeans_1d_dp = importr('Ckmeans.1d.dp')

from ckmeans import ckmeans

KmeansData = namedtuple('KmeansData', ['x', 'k'])
kmeans_data = {}

class TestCkmeans(TestCase):
    @classmethod
    def setUpClass(cls):
        gaussians = np.concatenate((rand.normal(3, 1/2, 15), rand.normal(23, 1/2, 13)))
        kmeans_data['gaussians'] = KmeansData(gaussians, (2, 3))

        values = np.array([ 176.2,  205.2,  206.2,  207.2,  226.2,  237.2,  239.2,  249.2,
            252.2,  254.2,  266.2,  269.2,  275.2,  276.2,  279.2,  296.2, 297.2,
            305.2,  306.2,  310.2])
        counts = np.array([  6,  14,  84,  20,  84, 280,   4,   2,  20,   4,   8,  12,
            42, 4,  52,  26,  30,  54,  12,  10])
        messy = np.concatenate([np.repeat(value, count) for value, count in zip(values, counts)])
        kmeans_data['guess_k'] = KmeansData(messy, (1, 10))

        # test copied from Ckmeans.1d.dp
        kmeans_data['given_k'] = KmeansData(np.array([-1, 2, -1, 2, 4, 5, 6, -1, 2, -1]), (3, 3))

        cls.kmeans_data = kmeans_data

    def assertArraysEqual(self, label, array1, array2, *args, **kwargs):
        self.assertSequenceEqual(array1.shape, array2.shape)
        np.testing.assert_allclose(array1, array2, *args, **kwargs, err_msg=label)

    def compare_results(self, my_result, r_result):
        self.assertArraysEqual(
            'clustering',
            my_result.clustering,
            np.asarray(r_result.rx2('cluster')) - 1
        )
        self.assertArraysEqual(
            'centers',
            my_result.centers,
            np.asarray(r_result.rx2('centers'))
        )
        self.assertArraysEqual(
            'within_ss',
            my_result.within_ss,
            np.asarray(r_result.rx2('withinss'))
        )
        self.assertArraysEqual(
            'sizes',
            my_result.sizes,
            np.asarray(r_result.rx2('size'))
        )
        self.assertAlmostEqual(my_result.total_ss, r_result.rx2('totss')[0]                )
        self.assertAlmostEqual(my_result.between_ss, r_result.rx2('betweenss')[0]            )

    def run_test(self, test):
        x, k = self.kmeans_data[test]
        my_result = ckmeans(x, k)
        r_result = Ckmeans_1d_dp.Ckmeans_1d_dp(x, np.array(k))
        self.compare_results(my_result, r_result)

    def test_given_k(self):
        self.run_test('given_k')

    def test_guess_k(self):
        self.run_test('guess_k')

    def test_gaussians(self):
        self.run_test('gaussians')
