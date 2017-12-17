import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy.stats as ss

from code import utils
from code import linear_regression_lvm
from code.linear_regression_lvm import compute_var


class TestLinearRegressionLVM(unittest.TestCase):

    def test_mle_expand_and_unpack(self):
        '''Check that expands batch, then unpacks computation correctly'''
        N = 5
        x = np.arange(N).astype(float).reshape(-1, 1)  # e.g., N instances with 1 element
        x_var = utils.make_torch_variable(x, False)

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
            beta=utils.make_torch_variable(beta, True),
            sigma=utils.make_torch_variable([sigma], True)
        )
        batch_var = utils.make_torch_variable(batch, False)
        noise_var = utils.make_torch_variable(noise, False)
        test_marginal_log_lik = linear_regression_lvm.mle_estimate_batch_likelihood(
            batch_var, mle_params, sub_B, test_noise=noise_var
        )

        assert_array_almost_equal(truth_marginal_log_lik, test_marginal_log_lik.data.numpy())

        # Check gradients
        test_marginal_log_lik.backward()
        self.assertIsNotNone(mle_params.beta.grad)
        self.assertIsNotNone(mle_params.sigma.grad)

    def test_compute_var(self):
        '''Make sure computing variance of posterior is autograd-able'''
        # Compute quantity
        beta = utils.make_torch_variable(np.array([[1, 1]]).astype(float), True)
        sigma = utils.make_torch_variable([1.0], True)
        var = compute_var(beta, sigma)

        # Check shape
        I, _ = var.size()
        self.assertEqual(2, I)

        # Check grad
        var.sum().sum().backward()
        self.assertIsNotNone(beta.grad)
        self.assertIsNotNone(sigma.grad)

    def test_estimate_batch_likelihood_v2(self):
        '''Test computation of marginal likelihood with latent var's marginalized out'''
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

        beta = np.array([[1, 0.1]]).astype(float)
        sigma = 1.0
        var = np.dot(beta.T, beta) + (sigma ** 2) * np.identity(M)

        truth_marginal_lik = ss.multivariate_normal.pdf(batch, np.zeros(M), var)
        truth_marginal_log_lik = np.log(truth_marginal_lik).sum()

        mle_params = linear_regression_lvm.MLE_PARAMS(
            beta=utils.make_torch_variable(beta, True),
            sigma=utils.make_torch_variable([sigma], True)
        )
        batch_var = utils.make_torch_variable(batch, False)
        test_marginal_log_lik = linear_regression_lvm.mle_estimate_batch_likelihood_v2(
            batch_var, mle_params
        )

        assert_array_almost_equal(truth_marginal_log_lik, test_marginal_log_lik.data.numpy())

        # Check gradients
        test_marginal_log_lik.backward()
        self.assertIsNotNone(mle_params.beta.grad)
        self.assertIsNotNone(mle_params.sigma.grad)

    def test_estimate_batch_likelihood_v3(self):
        '''Test computation of marginal likelihood with latent var's marginalized out'''
        # Implied dimensions: B = 3, M = 2, K = 1, sub_B = 2
        N = 3
        M = 2
        K = 1

        batch = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
        ]).astype(float)

        z = np.array([[0], [0], [0]]).astype(float)
        sigma = 1.0
        alpha = 1.0

        var = (alpha ** 2) * np.dot(z, z.T) + (sigma ** 2) * np.identity(N)

        # These are the computations used in Lawrence's paper
        # Serve as nice secondary verification that we are computing things correctly
        a_0 = N * M * np.log(2 * np.pi)
        a_1 = M * np.log(np.linalg.det(var))
        a_2 = np.diag(np.dot(np.linalg.pinv(var), np.dot(batch, batch.T))).sum()
        truth_marginal_log_lik = -0.5 * (a_0 + a_1 + a_2)

        mle_params_2 = linear_regression_lvm.MLE_PARAMS_2(
            z=utils.make_torch_variable(z, True),
            sigma=utils.make_torch_variable([sigma], True),
            alpha=utils.make_torch_variable([alpha], True)
        )
        batch_var = utils.make_torch_variable(batch, False)
        batch_ix = np.array([0, 1, 2])

        test_marginal_log_lik = linear_regression_lvm.mle_estimate_batch_likelihood_v3(
            batch_var, batch_ix, mle_params_2
        )
        test_marginal_log_lik = test_marginal_log_lik.sum()

        assert_array_almost_equal(truth_marginal_log_lik, test_marginal_log_lik.data.numpy())

        # Check gradients
        test_marginal_log_lik.backward()
        self.assertIsNotNone(mle_params_2.z.grad)
        self.assertIsNotNone(mle_params_2.sigma.grad)
        self.assertIsNotNone(mle_params_2.alpha.grad)

    def test_em_compute_posterior_and_extract_diagonals(self):
        '''Ensure EM code passes snuff'''
        # Implied dimensions: B = 3, M = 2, K = 1, sub_B = 2
        B = 3
        M = 2
        K = 2
        sub_B = 2

        batch = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
        ]).astype(float)

        beta = np.array([
            [1, 0.1],
            [1, 0.1]
        ]).astype(float)
        sigma = 1.0

        var_inv = np.linalg.pinv(np.dot(beta, beta.T) + (sigma ** 2) * np.identity(K))
        truth_e_z = np.dot(var_inv, np.dot(beta, batch.T)).T
        truth_e_z2 = np.empty((B, K, K))
        for i in range(B):  # e.g., compute covariance matrix for latent of each batch point...
            dot_prod = np.dot(truth_e_z[[i], :].T, truth_e_z[[i], :])
            truth_e_z2[i, :, :] = (sigma ** 2) * var_inv + dot_prod

        em_params = linear_regression_lvm.EM_PARAMS(
            beta=utils.make_torch_variable(beta, True),
            sigma=utils.make_torch_variable([sigma], True)
        )
        batch_var = utils.make_torch_variable(batch, False)
        test_e_z, test_e_z2 = linear_regression_lvm.em_compute_posterior(
            batch_var, em_params
        )

        # Compare posteriors
        assert_array_almost_equal(truth_e_z, test_e_z.data.numpy())
        assert_array_almost_equal(truth_e_z2, test_e_z2.data.numpy())

        # Check for gradients
        test_e_z.sum().backward(retain_graph=True)
        self.assertIsNotNone(em_params.beta.grad)
        self.assertIsNotNone(em_params.sigma.grad)

        em_params.beta.grad = None
        em_params.sigma.grad = None

        test_e_z2.sum().backward(retain_graph=True)
        self.assertIsNotNone(em_params.beta.grad)
        self.assertIsNotNone(em_params.sigma.grad)

        # Additionally, check that diagonal extraction step preserves gradients
        em_params.beta.grad = None
        em_params.sigma.grad = None

        truth_diagonals = np.empty((B, K))
        for i in range(B):
            truth_diagonals[i, :] = np.diag(truth_e_z2[i, :, :])

        test_diagonals = linear_regression_lvm.extract_diagonals(test_e_z2)

        assert_array_almost_equal(truth_diagonals, test_diagonals.data.numpy())

        test_diagonals.sum().backward(retain_graph=True)
        self.assertIsNotNone(em_params.beta.grad)
        self.assertIsNotNone(em_params.sigma.grad)

    def test_em_full_data_log_likelihood(self):
        # Implied dimensions: B = 3, M = 2, K = 1, sub_B = 2
        B = 3
        M = 2
        K = 2

        batch = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
        ]).astype(float)

        beta = np.array([
            [1, 0.1],
            [1, 0.1]
        ]).astype(float)
        sigma = 1.0

        var_inv = np.linalg.pinv(np.dot(beta, beta.T) + (sigma ** 2) * np.identity(K))
        truth_e_z = np.dot(var_inv, np.dot(beta, batch.T)).T
        truth_e_z2 = np.empty((B, K, K))
        for i in range(B):  # e.g., compute covariance matrix for latent of each batch point...
            dot_prod = np.dot(truth_e_z[[i], :].T, truth_e_z[[i], :])
            truth_e_z2[i, :, :] = (sigma ** 2) * var_inv + dot_prod

        truth_log_lik = np.empty((B, ))
        for i in range(B):
            a_1 = (M / 2.0) * np.log(sigma ** 2)
            a_2 = 0.5 * np.sum(np.diag(truth_e_z2[i, :, :]))
            a_3 = 0.5 * np.dot(batch[[i], :], batch[[i], :].T)
            a_4 = -1 * np.dot(truth_e_z[[i], :], np.dot(beta, batch[[i], :].T))
            a_5 = 0.5 * np.sum(np.diag(np.dot(beta, np.dot(beta.T, truth_e_z2[i, :, :]))))

            truth_log_lik[i] = a_1 + a_2 + a_3 + a_4 + a_5

        truth_log_lik = truth_log_lik.sum()

        em_params = linear_regression_lvm.EM_PARAMS(
            beta=utils.make_torch_variable(beta, True),
            sigma=utils.make_torch_variable([sigma], True)
        )
        batch_var = utils.make_torch_variable(batch, False)
        test_e_z, test_e_z2 = linear_regression_lvm.em_compute_posterior(
            batch_var, em_params
        )
        test_log_lik = linear_regression_lvm.em_compute_full_data_log_likelihood(
            batch_var, em_params, test_e_z, test_e_z2
        )

        # Compare computation
        assert_array_almost_equal(truth_log_lik, test_log_lik.data.numpy())

        # Check for grads
        test_log_lik.backward()
        self.assertIsNotNone(em_params.beta.grad)
        self.assertIsNotNone(em_params.sigma.grad)

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
            beta_q=utils.make_torch_variable(beta_q, True),
            sigma_q=utils.make_torch_variable(sigma_q, True),
        )

        batch_var = utils.make_torch_variable(batch, False)
        noise_var = utils.make_torch_variable(noise, False)
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
            beta=utils.make_torch_variable(beta, True),
            sigma=utils.make_torch_variable([sigma], True),
            beta_q=utils.make_torch_variable(beta_q, True),
            sigma_q=utils.make_torch_variable([sigma_q], True)
        )
        batch_var = utils.make_torch_variable(batch, False)
        noise_var = utils.make_torch_variable(noise, False)
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
