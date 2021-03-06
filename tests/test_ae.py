import unittest

import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import scipy.stats as ss
import torch

from code.autoencoder import Autoencoder
from code.utils import make_torch_variable
from code.variational_autoencoder import VAE, reparametrize_noise, vae_lower_bound, VAETest


class LinearTransformer(object):

    def __init__(self, weight_variable, bias_variable):
        '''Initialize transformer to imitate pytorch nn, but that just does a linear transform'''
        self.weights = weight_variable
        self.bias = bias_variable

        self.input_dim, self.output_dim = self.weights.size()

    def __call__(self, variable):
        return torch.add(torch.mm(variable, self.weights), self.bias)


class TestAutoencoder(unittest.TestCase):

    def test_autoencoder_class(self):
        N = 3
        M = 3
        K = 2

        x = np.ones((N, M)).astype(float)
        x_var = make_torch_variable(x, requires_grad=False)

        # ### Test output & check gradients when no hidden layers
        model = Autoencoder(M, K, [], [])

        # Ensure parameters correctly registered (there should be 2 weight + 2 bias = 4 tensors)
        parameters = list(model.parameters())

        num_parameters = len(parameters)
        self.assertEqual(4, num_parameters)

        weight_1 = np.ones((K, M)).astype(float)
        bias_1 = np.ones((K, )).astype(float)
        weight_2 = np.ones((M, K)).astype(float)
        bias_2 = np.ones((M, )).astype(float)

        parameters[0].data = torch.Tensor(weight_1).type(torch.FloatTensor)
        parameters[1].data = torch.Tensor(bias_1).type(torch.FloatTensor)
        parameters[2].data = torch.Tensor(weight_2).type(torch.FloatTensor)
        parameters[3].data = torch.Tensor(bias_2).type(torch.FloatTensor)

        # Ensure output matches
        truth = (np.dot(x, weight_1.T) + bias_1).clip(0)
        truth = np.dot(truth, weight_2.T) + bias_2

        test = model(x_var)

        assert_array_almost_equal(truth, test.data.numpy())

        # Check that gradients not none
        loss = ((x_var - test) ** 2).mean()
        loss.backward()
        for p in parameters:
            self.assertIsNotNone(p)

        # ### Test output & check gradients when hidden layers
        hidden = 4

        model = Autoencoder(M, K, [4], [4])

        # Ensure parameters correctly registered (there should be 4 weight + 4 bias = 8 tensors)
        parameters = list(model.parameters())

        num_parameters = len(parameters)
        self.assertEqual(8, num_parameters)

        weight_1 = np.ones((hidden, M)).astype(float)
        bias_1 = np.ones((hidden, )).astype(float)
        weight_2 = np.ones((K, hidden)).astype(float)
        bias_2 = np.ones((K, )).astype(float)
        weight_3 = np.ones((hidden, K)).astype(float)
        bias_3 = np.ones((hidden, )).astype(float)
        weight_4 = np.ones((M, hidden)).astype(float)
        bias_4 = np.ones((M, )).astype(float)

        parameters[0].data = torch.Tensor(weight_1).type(torch.FloatTensor)
        parameters[1].data = torch.Tensor(bias_1).type(torch.FloatTensor)
        parameters[2].data = torch.Tensor(weight_2).type(torch.FloatTensor)
        parameters[3].data = torch.Tensor(bias_2).type(torch.FloatTensor)
        parameters[4].data = torch.Tensor(weight_3).type(torch.FloatTensor)
        parameters[5].data = torch.Tensor(bias_3).type(torch.FloatTensor)
        parameters[6].data = torch.Tensor(weight_4).type(torch.FloatTensor)
        parameters[7].data = torch.Tensor(bias_4).type(torch.FloatTensor)

        # Ensure output matches
        truth = (np.dot(x, weight_1.T) + bias_1).clip(0)
        truth = (np.dot(truth, weight_2.T) + bias_2).clip(0)
        truth = (np.dot(truth, weight_3.T) + bias_3).clip(0)
        truth = np.dot(truth, weight_4.T) + bias_4

        test = model(x_var)

        assert_array_almost_equal(truth, test.data.numpy())

        # Check that gradients not none
        loss = ((x_var - test) ** 2).mean()
        loss.backward()
        for p in parameters:
            self.assertIsNotNone(p)

    def test_variational_autoencoder(self):
        '''Make sure calculations are as expected & that gradients preserved'''
        # Establish parameters
        n = 4
        m1 = 3
        m2 = 2

        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ])

        encoder_mu_weights = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ]) / 3.0
        encoder_mu_bias = np.array([0.0, 0.0])
        encoder_logvar_weights = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        encoder_logvar_bias = np.log(np.array([4.0, 4.0]))

        decoder_mu_weights = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ]) / 2.0
        decoder_mu_bias = np.array([0.0, 0.0, 0.0])
        decoder_logvar_weights = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        decoder_logvar_bias = np.log(np.array([4.0, 4.0, 4.0]))

        x_var = make_torch_variable(x, requires_grad=False)

        encoder_mu_weights_var = make_torch_variable(encoder_mu_weights, requires_grad=True)
        encoder_mu_bias_var = make_torch_variable(encoder_mu_bias, requires_grad=True)
        encoder_logvar_weights_var = make_torch_variable(encoder_logvar_weights, requires_grad=True)
        encoder_logvar_bias_var = make_torch_variable(encoder_logvar_bias, requires_grad=True)

        decoder_mu_weights_var = make_torch_variable(decoder_mu_weights, requires_grad=True)
        decoder_mu_bias_var = make_torch_variable(decoder_mu_bias, requires_grad=True)
        decoder_logvar_weights_var = make_torch_variable(decoder_logvar_weights, requires_grad=True)
        decoder_logvar_bias_var = make_torch_variable(decoder_logvar_bias, requires_grad=True)

        encoder_mu = LinearTransformer(encoder_mu_weights_var, encoder_mu_bias_var)
        encoder_logvar = LinearTransformer(encoder_logvar_weights_var, encoder_logvar_bias_var)
        decoder_mu = LinearTransformer(decoder_mu_weights_var, decoder_mu_bias_var)
        decoder_logvar = LinearTransformer(decoder_logvar_weights_var, decoder_logvar_bias_var)

        vae = VAETest(m1, m2, encoder_mu, encoder_logvar, decoder_mu, decoder_logvar,
                      use_sampling=False)

        # Test the reparametrize noise function

        noise = np.array([  # Dimension n x m2
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ])

        reparam_mu = np.dot(x, encoder_mu_weights)
        reparam_sigma = 2.0 * np.identity(m2)  # e.g., just the bias
        expected_reparam = np.dot(noise, reparam_sigma) + reparam_mu

        noise_var = make_torch_variable(noise, requires_grad=False)

        test_reparam = reparametrize_noise(x_var, noise_var, vae)

        assert_array_almost_equal(expected_reparam, test_reparam.data.numpy(), decimal=5)

        # Test the lower bound, treating noise as "z"

        z = expected_reparam

        x_mu = np.dot(z, decoder_mu_weights)
        x_sigma_2 = 1.0

        z_mu = np.dot(x, encoder_mu_weights)
        z_sigma_2 = np.exp(encoder_logvar_bias)

        eye_m1 = np.identity(m1)

        expected_bound = 0.0
        for i in range(n):
            log_posterior = -0.5 * (np.log(z_sigma_2) + 1 + np.log(2 * np.pi)).sum()
            log_likelihood = ss.multivariate_normal.logpdf(x[i, :], x_mu[i, :], x_sigma_2 * eye_m1)
            log_prior = -0.5 * ((z_mu[i, :]) ** 2 + z_sigma_2 + np.log(2 * np.pi)).sum()

            expected_bound += -1 * (log_posterior - log_likelihood - log_prior)

        z_var = make_torch_variable(z, requires_grad=False)

        test_bound_01 = vae_lower_bound(x_var, z_var, vae)
        test_bound_02 = vae.forward(x_var, noise=noise_var)

        assert_array_almost_equal(expected_bound, test_bound_01.data.numpy(), decimal=4)
        assert_array_almost_equal(expected_bound, test_bound_02.data.numpy(), decimal=4)

        # Check gradients
        test_bound_02.backward()
        self.assertIsNotNone(encoder_mu_weights_var.grad)
        self.assertIsNotNone(encoder_mu_bias_var.grad)
        self.assertIsNotNone(encoder_logvar_weights_var.grad)
        self.assertIsNotNone(encoder_logvar_bias_var.grad)
        self.assertIsNotNone(decoder_mu_weights_var.grad)
        self.assertIsNotNone(decoder_mu_bias_var.grad)
        # self.assertIsNotNone(decoder_logvar_weights_var.grad)
        # self.assertIsNotNone(decoder_logvar_bias_var.grad)
