import unittest

import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import scipy.stats as ss

from code.gaussian_process_lvm import rbf_kernel_forward, _make_cov, _mle_log_likelihood
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