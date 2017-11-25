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

    def test_mle_estimate_batch_likelihood(self):
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

    def test_vb_reparametrize_noise(self):
        N = 4
        M = 3
        K = 2
        batch = np.array([[i for _ in range(M)] for i in range(N)]).astype(float)
        noise = np.ones((N, K)).astype(float)
        beta_q = np.array([
            [1, 1],
            [1, 1],
            [1, 1]
        ]).astype(float)
        sigma_q = np.array([2]).astype(float)

        vb_params = linear_regression_lvm.VB_PARAMS(
            beta=None,
            sigma=None,
            beta_q=linear_regression_lvm.make_torch_variable(beta_q, True),
            sigma_q=linear_regression_lvm.make_torch_variable(sigma_q, True),
        )

        batch_var = linear_regression_lvm.make_torch_variable(batch, False)
        noise_var = linear_regression_lvm.make_torch_variable(noise, False)
        reparam_noise = linear_regression_lvm._reparametrize_noise(batch_var, noise_var, vb_params)

        # Check values
        truth = np.array([[4 + 3 * i] * K for i in range(N)]).astype(float)
        assert_array_almost_equal(truth, reparam_noise.data.numpy())

        # Check gradients
        reparam_noise.sum().sum().backward()
        self.assertIsNotNone(vb_params.beta_q.grad)
        self.assertIsNotNone(vb_params.sigma_q.grad)

    def test_vb_estimate_lower_bound(self):
        # Implied dimensions: B = 3, M = 2, K = 1, sub_B = 2
        B = 3
        M = 2
        K = 1

        batch = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
        ]).astype(float)

        noise = np.array([
            [1],
            [1],
            [1]
        ]).astype(float)

        beta = np.array([[1, 1]]).astype(float)
        sigma = 1.0
        beta_q = np.array([[1], [1]]).astype(float)
        sigma_q = 1.0

        # e.g., noise * beta
        mu_x = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
        ]).astype(float)

        # e.g., batch * beta_q
        mu_q = np.array([
            [0],
            [2],
            [4],
        ]).astype(float)

        diff_x = batch - mu_x
        diff_q = noise - mu_q

        likelihood = ss.multivariate_normal.pdf(diff_x, np.zeros(M), sigma * np.identity(M))
        posterior = ss.multivariate_normal.pdf(diff_q, np.zeros(K), sigma * np.identity(K))
        prior = ss.multivariate_normal.pdf(noise, np.zeros(K), np.identity(K))

        truth_lower_bound = (np.log(posterior) - np.log(likelihood) - np.log(prior)).sum()

        vb_params = linear_regression_lvm.VB_PARAMS(
            beta=linear_regression_lvm.make_torch_variable(beta, True),
            sigma=linear_regression_lvm.make_torch_variable([sigma], True),
            beta_q=linear_regression_lvm.make_torch_variable(beta_q, True),
            sigma_q=linear_regression_lvm.make_torch_variable([sigma_q], True)
        )
        batch_var = linear_regression_lvm.make_torch_variable(batch, False)
        noise_var = linear_regression_lvm.make_torch_variable(noise, False)
        test_lower_bound = linear_regression_lvm.vb_estimate_lower_bound(
            batch_var, noise_var, vb_params
        )

        assert_array_almost_equal(truth_lower_bound, test_lower_bound.data.numpy())

        # Check gradients
        test_lower_bound.backward()
        self.assertIsNotNone(vb_params.beta.grad)
        self.assertIsNotNone(vb_params.sigma.grad)
        self.assertIsNotNone(vb_params.beta_q.grad)
        self.assertIsNotNone(vb_params.sigma_q.grad)
