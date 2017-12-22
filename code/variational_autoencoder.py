import numpy as np
import torch
from torch import nn

from code.mvn import torch_mvn_density
from code.utils import make_torch_variable


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

    def forward(self, x, noise=None):
        n, _ = x.size()

        if noise is None:
            noise = make_torch_variable(np.random.randn(n, self.k), requires_grad=False)

        sample_z = reparametrize_noise(x, noise, self)

        return vae_lower_bound(x, sample_z, self)


def reparametrize_noise(x, gaussian_noise, vae_model):
    mu = vae_model.encoder_mu(x)
    sigma = vae_model.encoder_sigma(x)

    reparametrized = torch.add(mu, torch.mul(sigma ** 2, gaussian_noise))

    return reparametrized


def _expand_batch_sigma(sigma, m):
    B, _ = sigma.size()

    sigma = sigma.unsqueeze(2)  # B x 1 x 1

    a = make_torch_variable(np.ones((m, 1)), requires_grad=False).expand(B, m, 1)
    b = make_torch_variable(np.ones((1, m)), requires_grad=False).expand(B, 1, m)
    c = make_torch_variable(np.identity(m), requires_grad=False).expand(B, m, m)

    # [(B x 3 x 1) x (B x 1 x 1)] x (B x 1 x 3) element-wise (B x 3 x 3) --> B x
    cov = torch.bmm(torch.bmm(a, sigma ** 2), b) * c

    return cov


def vae_lower_bound(x, z, vae_model):
    '''Compute variational lower bound, as specified by variational autoencoder'''
    # Some initial parameter setting
    n, m1 = x.size()
    _, m2 = z.size()

    # Parameters of the likelihood of x given the model & z
    x_mu = vae_model.decoder_mu(z)  # b x m1
    x_sigma = _expand_batch_sigma(vae_model.decoder_sigma(z) ** 2, m1)  # b x m1 x m1

    # Parameters of the variational approximation of the posterior of z given model & x
    z_mu = vae_model.encoder_mu(x)  # b x m2
    z_sigma = _expand_batch_sigma(vae_model.encoder_sigma(x) ** 2, m2)  # b x m2 x m2

    # Parameters of the prior of z
    prior_mu = make_torch_variable(np.zeros(m2), requires_grad=False)  # dim: m2
    prior_sigma = make_torch_variable(np.identity(m2), requires_grad=False)  # m2 x m2

    x_big = x.unsqueeze(1)  # reshape to n x 1 x m1
    z_big = z.unsqueeze(1)  # reshape to n x 1 x m2

    # Compute components
    lower_bound = make_torch_variable([0.0], requires_grad=False)
    for i in range(n):
        log_posterior = torch_mvn_density(z_big[i, :, :], z_mu[i, :], z_sigma[i, :, :], log=True)
        log_likelihood = torch_mvn_density(x_big[i, :, :], x_mu[i, :], x_sigma[i, :, :], log=True)
        log_prior = torch_mvn_density(z_big[i, :, :], prior_mu, prior_sigma, log=True)

        lower_bound += log_posterior - log_likelihood - log_prior

    return lower_bound


