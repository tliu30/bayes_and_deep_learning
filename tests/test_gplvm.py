import unittest

import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import scipy.stats as ss

from code.gaussian_process_lvm import (
    rbf_kernel_forward,
    _make_cov,
    _mle_log_likelihood,
    MLE_PARAMS,
    mle_batch_log_likelihood,
    _gp_conditional_mean_cov,
    _inactive_point_likelihood, _vb_lower_bound, _reparametrize_noise)
from code.utils import make_torch_variable


class TestGPLVM(unittest.TestCase):

    def test_rbf_kernel_forward(self):
        '''Make sure we know the RBF kernel is working'''
        # Test basic functionality
        x1 = np.array([0.0, 0.0])
        x2 = np.array([1.0, 1.0])
        log_l = np.log(2.0)
        eps = 1e-5

        sq_dist = ((x2 - x1) ** 2).sum()
        expected = np.exp(-1.0 / np.exp(log_l) * sq_dist)

        x1_var = make_torch_variable([x1], requires_grad=False)
        x2_var = make_torch_variable([x2], requires_grad=False)
        log_l_var = make_torch_variable([log_l], requires_grad=True)

        test = rbf_kernel_forward(x1_var, x2_var, log_l_var, eps=eps)

        assert_array_almost_equal(expected, test.data.numpy()[0, 0], decimal=5)

        # Make sure the gradient gets through

        test.sum().backward()
        self.assertIsNotNone(log_l_var.grad)

        # Test safety valve

        bad_log_l = -1e6

        expected_bad = np.exp(-1.0 / eps * sq_dist)

        bad_log_l_var = make_torch_variable([bad_log_l], requires_grad=True)

        test_bad = rbf_kernel_forward(x1_var, x2_var, bad_log_l_var, eps=eps)

        assert_array_almost_equal(expected_bad, test_bad.data.numpy()[0, 0], decimal=5)

        # Make sure the gradient gets through

        test_bad.sum().backward()
        self.assertIsNotNone(bad_log_l_var.grad)

    def test_make_cov(self):
        '''Make sure the covariance function is working'''
        # Check value
        x1 = np.array([0.0, 0.0])
        x2 = np.array([1.0, 1.0])
        alpha = 2.0
        sigma = 2.0
        log_l = np.log(2.0)

        sq_dist = ((x2 - x1) ** 2).sum()
        rbf = np.exp(-1.0 / np.exp(log_l) * sq_dist)
        expected_cov = (alpha ** 2) * rbf + sigma ** 2

        x1_var = make_torch_variable([x1], requires_grad=False)
        x2_var = make_torch_variable([x2], requires_grad=False)
        alpha_var = make_torch_variable([alpha], requires_grad=True)
        sigma_var = make_torch_variable([sigma], requires_grad=True)
        log_l_var = make_torch_variable([log_l], requires_grad=True)

        test_cov = _make_cov(x1_var, x2_var, alpha_var, sigma_var, log_l_var)

        assert_array_almost_equal(expected_cov, test_cov.data.numpy()[0, 0], decimal=5)

        # Make sure the gradient gets through

        test_cov.sum().backward()
        self.assertIsNotNone(alpha_var.grad)
        self.assertIsNotNone(sigma_var.grad)
        self.assertIsNotNone(log_l_var.grad)

    def test_mle_log_likelihood(self):
        '''Check validity of computation'''
        # Check values
        x = np.array([
            [0.0, 0.0],
            [1.0, 1.0]
        ])
        z = np.array([
            [0.0],
            [1.0]
        ])
        alpha = 2.0
        sigma = 2.0
        log_l = np.log(2.0)

        sq_dist_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        rbf_kernel = np.exp(-1.0 / np.exp(log_l) * sq_dist_matrix)
        cov = (alpha ** 2) * rbf_kernel + (sigma ** 2) * np.identity(2)

        mu = np.array([0.0, 0.0])

        expected = np.sum(np.log(ss.multivariate_normal(mean=mu, cov=cov).pdf(x.T)))

        x_var = make_torch_variable(x, requires_grad=False)
        z_var = make_torch_variable(z, requires_grad=False)
        alpha_var = make_torch_variable([alpha], requires_grad=True)
        sigma_var = make_torch_variable([sigma], requires_grad=True)
        log_l_var = make_torch_variable([log_l], requires_grad=True)

        test = _mle_log_likelihood(x_var, z_var, alpha_var, sigma_var, log_l_var)

        assert_array_almost_equal(expected, test.data.numpy()[0], decimal=5)

        # Check gradients
        test.backward()
        self.assertIsNotNone(alpha_var.grad)
        self.assertIsNotNone(sigma_var.grad)
        self.assertIsNotNone(log_l_var.grad)

        # Check that wrapper function also preserves gradients
        x_var = make_torch_variable(x, requires_grad=False)
        mle_params = MLE_PARAMS(
            z=make_torch_variable(z, requires_grad=False),
            alpha=make_torch_variable([alpha], requires_grad=True),
            sigma=make_torch_variable([sigma], requires_grad=True),
            log_l=make_torch_variable([log_l], requires_grad=True),
        )
        batch_ix = np.array([0, 1])

        test = mle_batch_log_likelihood(x_var, mle_params, batch_ix)

        assert_array_almost_equal(expected, test.data.numpy()[0], decimal=5)

        test.backward()
        self.assertIsNotNone(mle_params.alpha.grad)
        self.assertIsNotNone(mle_params.sigma.grad)
        self.assertIsNotNone(mle_params.log_l.grad)

    def test_inactive_active_likelihood(self):
        # ### Check computation of E[x | z, active set] and Var(x | z, active set)

        # Compute the active set's covariance
        active_x = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        active_z = np.array([
            [0.0],
            [1.0],
            [2.0],
        ])
        alpha = 2.0
        sigma = 2.0
        log_l = np.log(2.0)

        active_sq_dist_matrix = np.array([
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
        ])
        active_rbf_kernel = np.exp(-1.0 / np.exp(log_l) * active_sq_dist_matrix)
        active_cov = (alpha ** 2) * active_rbf_kernel + (sigma ** 2) * np.identity(3)
        inv_active_cov = np.linalg.pinv(active_cov)

        # Compute values relative to inactive set
        inactive_z = np.array([
            [0.5]
        ])

        inactive_sq_dist_matrix = np.array([
            [0.5 ** 2],
            [0.5 ** 2],
            [1.5 ** 2],
        ])
        print(inactive_sq_dist_matrix)
        inactive_rbf_kernel = np.exp(-1.0 / np.exp(log_l) * inactive_sq_dist_matrix)
        cross_cov = (alpha ** 2) * inactive_rbf_kernel
        inactive_var = (alpha ** 2) + (sigma ** 2)

        # Compute the true values
        expected_mu = np.dot(active_x.T, np.dot(inv_active_cov, cross_cov)).T
        expected_var = inactive_var - np.dot(cross_cov.T, np.dot(inv_active_cov, cross_cov))
        expected_sigma = expected_var * np.identity(2)

        # Compute the test values
        active_x_var = make_torch_variable(active_x, requires_grad=False)
        active_z_var = make_torch_variable(active_z, requires_grad=False)
        alpha_var = make_torch_variable([alpha], requires_grad=False)
        sigma_var = make_torch_variable([sigma], requires_grad=False)
        log_l_var = make_torch_variable([log_l], requires_grad=False)
        inactive_z_var = make_torch_variable(inactive_z, requires_grad=True)
        test_mu, test_sigma = _gp_conditional_mean_cov(
            inactive_z_var, active_x_var, active_z_var, alpha_var, sigma_var, log_l_var
        )

        assert_array_almost_equal(expected_mu, test_mu.data.numpy(), decimal=5)
        assert_array_almost_equal(expected_sigma, test_sigma.data.numpy(), decimal=5)

        # Check gradient
        test_mu.sum().backward(retain_graph=True)
        self.assertIsNotNone(inactive_z_var.grad)
        inactive_z_var.grad = None

        test_sigma.sum().backward(retain_graph=True)
        self.assertIsNotNone(inactive_z_var.grad)

        # ### Next, check computation of likelihood

        # Double the num points to check loop works
        inactive_x = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        inactive_z = np.array([
            [0.5],
            [0.5]
        ])

        likelihood = ss.multivariate_normal(mean=expected_mu[0, :], cov=expected_sigma)
        expected_log_lik = 2 * likelihood.logpdf(np.array([0.0, 0.0]))

        inactive_x_var = make_torch_variable(inactive_x, requires_grad=False)
        inactive_z_var = make_torch_variable(inactive_z, requires_grad=True)
        test_log_lik = _inactive_point_likelihood(
            active_x_var, active_z_var, inactive_x_var, inactive_z_var,
            alpha_var, sigma_var, log_l_var
        )

        assert_array_almost_equal(expected_log_lik, test_log_lik.data.numpy(), decimal=5)

        # Check grad
        test_log_lik.sum().backward()
        self.assertIsNotNone(inactive_z_var.grad)

    def test_vb_lower_bound(self):
        '''Check computations'''
        # Establish parameters
        x = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        z = np.array([
            [0.0],
            [1.0],
            [2.0],
        ])

        alpha = 2.0
        sigma = 2.0
        log_l = np.log(2.0)

        alpha_q = 2.0
        sigma_q = 2.0
        log_l_q = np.log(2.0)

        # Compute covariances
        sq_dist_x = np.array([
            [0.0, 2.0, 8.0],
            [2.0, 0.0, 2.0],
            [8.0, 2.0, 0.0],
        ])
        rbf_x = np.exp(-1.0 / np.exp(log_l_q) * sq_dist_x)
        cov_z = (alpha_q ** 2) * rbf_x + (sigma_q ** 2) * np.identity(3)

        sq_dist_z = np.array([
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
        ])
        rbf_z = np.exp(-1.0 / np.exp(log_l) * sq_dist_z)
        cov_x = (alpha ** 2) * rbf_z + (sigma ** 2) * np.identity(3)

        # Compute components of bound
        log_posterior = ss.multivariate_normal.logpdf(z.T, mean=np.zeros(3), cov=cov_z).sum()
        log_likelihood = ss.multivariate_normal.logpdf(x.T, mean=np.zeros(3), cov=cov_x).sum()
        log_prior = ss.multivariate_normal.logpdf(z, mean=np.zeros(1), cov=np.identity(1)).sum()

        expected_lower_bound = log_posterior - log_likelihood - log_prior

        # Compute test value
        x_var = make_torch_variable(x, requires_grad=False)
        z_var = make_torch_variable(z, requires_grad=False)
        alpha_var = make_torch_variable([alpha], requires_grad=True)
        sigma_var = make_torch_variable([sigma], requires_grad=True)
        log_l_var = make_torch_variable([log_l], requires_grad=True)
        alpha_q_var = make_torch_variable([alpha_q], requires_grad=True)
        sigma_q_var = make_torch_variable([sigma_q], requires_grad=True)
        log_l_q_var = make_torch_variable([log_l_q], requires_grad=True)
        test_lower_bound = _vb_lower_bound(
            x_var, z_var, alpha_var, sigma_var, log_l_var, alpha_q_var, sigma_q_var, log_l_q_var
        )

        assert_array_almost_equal(expected_lower_bound, test_lower_bound.data.numpy(), decimal=5)

        # ### Check gradients
        test_lower_bound.backward()
        self.assertIsNotNone(alpha_var.grad)
        self.assertIsNotNone(sigma_var.grad)
        self.assertIsNotNone(log_l_var.grad)
        self.assertIsNotNone(alpha_q_var.grad)
        self.assertIsNotNone(sigma_q_var.grad)
        self.assertIsNotNone(log_l_q_var.grad)

    def test_vb_sample_noise(self):
        '''Check computations'''
        active_x = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        active_z = np.array([
            [0.0],
            [1.0],
            [2.0],
        ])
        alpha_q = 2.0
        sigma_q = 2.0
        log_l_q = np.log(2.0)

        active_sq_dist_matrix = np.array([
            [0.0, 2.0, 8.0],
            [2.0, 0.0, 2.0],
            [8.0, 2.0, 0.0],
        ])
        active_rbf_kernel = np.exp(-1.0 / np.exp(log_l_q) * active_sq_dist_matrix)
        active_cov = (alpha_q ** 2) * active_rbf_kernel + (sigma_q ** 2) * np.identity(3)
        inv_active_cov = np.linalg.pinv(active_cov)

        # Compute values relative to inactive set
        inactive_x = np.array([
            [0.5, 0.5]
        ])

        inactive_sq_dist_matrix = np.array([
            [2 * (0.5 ** 2)],
            [2 * (0.5 ** 2)],
            [2 * (1.5 ** 2)],
        ])
        inactive_rbf_kernel = np.exp(-1.0 / np.exp(log_l_q) * inactive_sq_dist_matrix)
        cross_cov = (alpha_q ** 2) * inactive_rbf_kernel
        inactive_var = (alpha_q ** 2) + (sigma_q ** 2)

        # Compute the true values
        expected_mu = np.dot(active_z.T, np.dot(inv_active_cov, cross_cov)).T
        expected_var = inactive_var - np.dot(cross_cov.T, np.dot(inv_active_cov, cross_cov))
        expected_sigma = expected_var * np.identity(2)

        inactive_noise = np.array([
            [0.0],
            [1.0]
        ])
        expected_reparam = inactive_noise * expected_sigma + expected_mu

        # Compute the test values
        active_x_var = make_torch_variable(active_x, requires_grad=False)
        active_z_var = make_torch_variable(active_z, requires_grad=False)
        alpha_q_var = make_torch_variable([alpha_q], requires_grad=False)
        sigma_q_var = make_torch_variable([sigma_q], requires_grad=False)
        log_l_q_var = make_torch_variable([log_l_q], requires_grad=False)
        inactive_x_var = make_torch_variable(inactive_x, requires_grad=True)
        inactive_noise_var = make_torch_variable(inactive_noise, requires_grad=True)
        test_reparam = _reparametrize_noise(
            inactive_x_var, inactive_noise_var, active_x_var, active_z_var,
            alpha_q_var, sigma_q_var, log_l_q_var
        )

        assert_array_almost_equal(expected_reparam, test_reparam, decimal=5)

        # Check gradient
        test_reparam.sum().backward()
        self.assertIsNotNone(alpha_q_var.grad)
        self.assertIsNotNone(sigma_q_var.grad)
        self.assertIsNotNone(log_l_q_var.grad)
