import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy.stats as ss

from code import linear_regression_lvm


class TestLinearRegressionLVM(unittest.TestCase):

    def test_mle_expand_and_unpack(self):
        '''Check that expands batch, then unpacks computation correctly'''
        N = 5
        x = np.arange(N).astype(float).reshape(-1, 1)  # e.g., N instances with 1 element
        x_var = linear_regression_lvm.make_torch_variable(x, False)

        reps = 3
        test_expand = linear_regression_lvm._mle_expand_batch(x_var, reps)  # e.g., (3 * N, 1)
        test_unpack = linear_regression_lvm._mle_unpack_likelihood(test_expand.squeeze(), reps, N)

        truth_expand = np.array([[i for _ in range(reps) for i in range(N)]]).astype(float).T
        truth_unpack = np.array([range(N) for _ in range(reps)]).astype(float)

        assert_array_almost_equal(np.array([reps * N, 1]), test_expand.size())
        assert_array_almost_equal(np.array([reps, N]), test_unpack.size())

        assert_array_almost_equal(truth_expand, test_expand.data.numpy())
        assert_array_almost_equal(truth_unpack, test_unpack.data.numpy())

    def test_estimate_batch_likelihood(self):
        '''Check that the batch likelihood creates correct calculations & leads to gradients'''
        # Implied dimensions: B = 3, M = 2, K = 1, sub_B = 2
        B = 3
        M = 2
        K = 1
        sub_B = 2

        batch = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
        ]).astype(float)

        noise = np.array([
            [-1],
            [-1],
            [-1],
            [1],
            [1],
            [1]
        ]).astype(float)

        beta = np.array([[1, 1]]).astype(float)
        sigma = 1.0

        expanded_batch = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [0, 0],
            [1, 1],
            [2, 2],
        ]).astype(float)

        iter_mu = np.array([
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [1, 1],
            [1, 1],
            [1, 1],
        ]).astype(float)

        diff = expanded_batch - iter_mu
        likelihoods = ss.multivariate_normal.pdf(diff, np.zeros(M), sigma * np.identity(M))
        expected_each_iter = np.array([
            (likelihoods[0] + likelihoods[3]) / 2,
            (likelihoods[1] + likelihoods[4]) / 2,
            (likelihoods[2] + likelihoods[5]) / 2,
        ])
        truth_marginal_log_lik = np.log(expected_each_iter).sum()

        mle_params = linear_regression_lvm.MLE_PARAMS(
            beta=linear_regression_lvm.make_torch_variable(beta, True),
            sigma=linear_regression_lvm.make_torch_variable([sigma], True)
        )
        batch_var = linear_regression_lvm.make_torch_variable(batch, False)
        noise_var = linear_regression_lvm.make_torch_variable(noise, False)
        test_marginal_log_lik = linear_regression_lvm.mle_estimate_batch_likelihood(
            batch_var, mle_params, sub_B, test_noise=noise_var
        )

        assert_array_almost_equal(truth_marginal_log_lik, test_marginal_log_lik.data.numpy())

        # Check gradients
        test_marginal_log_lik.backward()
        self.assertIsNotNone(mle_params.beta.grad)
        self.assertIsNotNone(mle_params.sigma.grad)
