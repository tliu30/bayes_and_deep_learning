import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy.stats as ss
import torch
from torch.autograd import Variable

from code import mvn
from code.utils import make_torch_variable


class TestMVN(unittest.TestCase):

    def test_determinant(self):
        '''Check that we correctly compute determinants, and the operation is differentiable'''
        # Initialize values
        sigma = np.array([
            [1,   0,   0.1],
            [0,   1,   0],
            [0.1, 0,   1]
        ]).astype(float)
        sigma_var = Variable(torch.Tensor(sigma).type(torch.FloatTensor), requires_grad=True)

        # Construct truth and test
        truth = np.linalg.det(sigma)
        test = mvn.torch_determinant(sigma_var)

        assert_array_almost_equal(truth, test.data.numpy())

        # Check availability of gradient
        test = mvn.torch_determinant(sigma_var)
        test.backward()
        self.assertIsNotNone(sigma_var.grad)

    def test_mvn_density(self):
        '''Check computation of multivariate normal densities'''
        # Initialize an (N, M), with N = 2, M = 3
        x = np.array([[0, 0, 0], [1, 1, 1]]).astype(float)

        # Create one mu shape for each accepted shape
        mu_0 = np.array([0.1, 0.1, 0.1]).astype(float)
        mu_1 = np.array([[0.1, 0.1, 0.1]]).astype(float)
        mu_2 = np.array([
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1]
        ]).astype(float)

        # Create sigma
        sigma = np.array([
            [1,   0,   0.1],
            [0,   1,   0],
            [0.1, 0,   1]
        ]).astype(float)

        # Convert types
        dtype = torch.FloatTensor
        x_var = Variable(torch.Tensor(x).type(dtype), requires_grad=False)
        mu_0_var = Variable(torch.Tensor(mu_0).type(dtype), requires_grad=True)
        mu_1_var = Variable(torch.Tensor(mu_1).type(dtype), requires_grad=False)
        mu_2_var = Variable(torch.Tensor(mu_2).type(dtype), requires_grad=False)
        sigma_var = Variable(torch.Tensor(sigma).type(dtype), requires_grad=True)

        # Create truth
        truth = ss.multivariate_normal.pdf(x, mean=mu_0, cov=sigma)
        test_0 = mvn.torch_mvn_density(x_var, mu_0_var, sigma_var)
        test_1 = mvn.torch_mvn_density(x_var, mu_1_var, sigma_var)
        test_2 = mvn.torch_mvn_density(x_var, mu_2_var, sigma_var)

        assert_array_almost_equal(truth, test_0.data.numpy())
        assert_array_almost_equal(truth, test_1.data.numpy())
        assert_array_almost_equal(truth, test_2.data.numpy())

        # Do the same for the log version
        log_truth = np.log(truth)
        log_test_0 = mvn.torch_mvn_density(x_var, mu_0_var, sigma_var, log=True)
        log_test_1 = mvn.torch_mvn_density(x_var, mu_1_var, sigma_var, log=True)
        log_test_2 = mvn.torch_mvn_density(x_var, mu_2_var, sigma_var, log=True)

        assert_array_almost_equal(log_truth, log_test_0.data.numpy())
        assert_array_almost_equal(log_truth, log_test_1.data.numpy())
        assert_array_almost_equal(log_truth, log_test_2.data.numpy())

        # Ensure there is a gradient
        mu_0_var = Variable(torch.Tensor(mu_0).type(dtype), requires_grad=True)
        sigma_var = Variable(torch.Tensor(sigma).type(dtype), requires_grad=True)
        test_0 = mvn.torch_mvn_density(x_var, mu_0_var, sigma_var)
        test_0 = test_0.prod()
        test_0.backward()
        self.assertIsNotNone(mu_0_var.grad)
        self.assertIsNotNone(sigma_var.grad)

        mu_0_var = Variable(torch.Tensor(mu_0).type(dtype), requires_grad=True)
        sigma_var = Variable(torch.Tensor(sigma).type(dtype), requires_grad=True)
        log_test_0 = mvn.torch_mvn_density(x_var, mu_0_var, sigma_var, log=True)
        log_test_0 = log_test_0.sum()
        log_test_0.backward()
        self.assertIsNotNone(mu_0_var.grad)
        self.assertIsNotNone(sigma_var.grad)

    def test_batch_mvn_computation(self):
        # Make sure diagonal util works as expected
        M = 5
        diags = np.array([1.0, 2.0, 3.0])
        expected_matrices = np.concatenate(
            [d * np.identity(M).reshape(1, M, M) for d in diags],
            axis=0
        )

        diags_var = Variable(torch.Tensor(diags), requires_grad=True)
        test_matrices = mvn._make_batch_matrices_w_diagonal(diags_var, M)

        assert_array_almost_equal(expected_matrices, test_matrices.data.numpy())

        # Make sure likelihood computation is correct
        B = 2
        M = 3

        x = np.array([  # shape (B, M)
            [0, 0, 0],
            [1, 1, 1]
        ]).astype(float)

        mu = np.array([  # shape (B, M)
            [0, 0, 0],
            [1, 1, 1]
        ]).astype(float)

        sigma = np.array([2, 3]).astype(float)  # shape (B, )

        cov = np.concatenate([  # shape(B, M, M)
            np.identity(M).reshape(1, M, M) * x for x in sigma
        ]) ** 2

        # Expected
        expected = np.array([  # shape (B, 1)
            ss.multivariate_normal.logpdf(x[i, :], mu[i, :], cov[i, :, :])
            for i in range(B)
        ])

        # Test
        x_var = make_torch_variable(x, requires_grad=True)
        mu_var = make_torch_variable(mu, requires_grad=True)
        sigma_var = make_torch_variable(sigma, requires_grad=True)
        test = mvn.torch_diagonal_mvn_density_batch(x_var, mu_var, sigma_var, log=True)

        assert_array_almost_equal(expected, test.data.numpy())

        # Check gradients
        test.sum().backward()
        self.assertIsNotNone(x_var.grad)
        self.assertIsNotNone(mu_var.grad)
        self.assertIsNotNone(sigma_var.grad)
