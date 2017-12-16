import numpy as np
import torch
from torch import nn

from code.linear_regression_lvm import make_torch_variable
from code.mvn import torch_mvn_density


class FullyConnectedNN(nn.Module):

    def __init__(self, layer_dimensions, hidden_activation_func, output_activation_func):
        super(FullyConnectedNN, self).__init__()
        layer_pairs = zip(layer_dimensions[:-1], layer_dimensions[1:])
        self.layers = nn.ModuleList([nn.Linear(n_in, n_out) for n_in, n_out in layer_pairs])
        self.hidden_activation_func = hidden_activation_func
        self.output_activation_func = output_activation_func

        self.input_dim = layer_dimensions[0]
        self.output_dim = layer_dimensions[-1]

    def forward(self, x):
        input = x
        for layer in self.layers[:-1]:
            input = self.hidden_activation_func(layer(input))
        out = self.output_activation_func(self.layers[-1](input))
        return out


class VAE(nn.Module):

    def __init__(self, m, k, encoder_mu, encoder_sigma, decoder_mu, decoder_sigma):
        super(VAE, self).__init__()

        # check shapes
        if m != encoder_mu.input_dim:
            raise ValueError(u'Bad shape')

        if k != encoder_mu.output_dim:
            raise ValueError(u'Bad shape')

        if k != decoder_mu.input_dim:
            raise ValueError(u'Bad shape')

        if m != decoder_mu.output_dim:
            raise ValueError(u'Bad shape')

        if m != encoder_sigma.input_dim:
            raise ValueError(u'Bad shape')

        if 1 != encoder_sigma.output_dim:
            raise ValueError(u'Bad shape')

        if k != decoder_sigma.input_dim:
            raise ValueError(u'Bad shape')

        if 1 != decoder_sigma.output_dim:
            raise ValueError(u'Bad shape')

        self.encoder_mu = encoder_mu
        self.encoder_sigma = encoder_sigma
        self.decoder_mu = decoder_mu
        self.decoder_sigma = decoder_sigma

        self.m = m
        self.k = k

    def forward(self, x):
        n, _ = x.size()

        noise = make_torch_variable(np.random.randn(n, self.k), requires_grad=False)
        sample_z = reparametrize_noise(noise, self)

        return vae_lower_bound(x, sample_z, self)


def reparametrize_noise(gaussian_noise, vae_model):
    mu = vae_model.encoder_mu(gaussian_noise)
    sigma = vae_model.encoder_sigma(gaussian_noise)

    reparametrized = torch.add(mu, torch.mul(sigma, gaussian_noise))

    return reparametrized


def vae_lower_bound(x, z, vae_model):
    '''Compute variational lower bound, as specified by variational autoencoder'''
    # Some initial parameter setting
    _, m1 = x.size()
    _, m2 = z.size()

    m1_m1_identity = make_torch_variable(np.identity(m1, m1), requires_grad=False)
    m2_m2_identity = make_torch_variable(np.identity(m2, m2), requires_grad=False)

    # Parameters of the likelihood of x given the model & z
    x_mu = vae_model.decoder_mu(z)
    x_sigma = torch.mul(vae_model.decoder_sigma(z) ** 2, m1_m1_identity)

    # Parameters of the variational approximation of the posterior of z given model & x
    z_mu = vae_model.encoder_mu(x)
    z_sigma = torch.mul(vae_model.encoder_sigma(x) ** 2, m2_m2_identity)

    # Parameters of the prior of z
    prior_mu = torch_mvn_density(np.zeros(m2), requires_grad=False)
    prior_sigma = m2_m2_identity

    # Compute components
    log_posterior = torch_mvn_density(z, z_mu, z_sigma, log=True)
    log_likelihood = torch_mvn_density(x, x_mu, x_sigma, log=True)
    log_prior = torch_mvn_density(z, prior_mu, prior_sigma, log=True)

    lower_bound = log_posterior - log_likelihood - log_prior

    return lower_bound.sum()


