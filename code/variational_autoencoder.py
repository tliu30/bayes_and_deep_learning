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
        '''Construct a variational autoencoder by providing encoder and decoder networks

        Remember that "encoders" parametrize the variational approximation to the posterior
            q(z | x) = MVN(encoder_mu(x), (encoder_sigma(x) ** 2) * I)
        and the "decoders" parametrize the likelihood
            p(x | z) = MVN(decoder_mu(z), (decoder_sigma(z) ** 2) * I)

        Args:
            m: (int) dimension of observations
            k: (int) dimension of latent variables
            encoder_mu: (nn.Module) n x m--> n x k; specifies posterior's location parameter
            encoder_sigma: (nn.Module) n x m--> n x 1; specifies posterior's scale parameter
            decoder_mu: (nn.Module) n x k --> n x m; specifies likelihood's location parameter
            decoder_sigma: (nn.Module) n x k--> n x 1; specifies likelihood's scale parameter
        '''
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
        '''Estimate variational lower bound of parameters given observations x'''
        n, _ = x.size()

        if noise is None:
            noise = make_torch_variable(np.random.randn(n, self.k), requires_grad=False)

        sample_z = reparametrize_noise(x, noise, self)

        return vae_lower_bound(x, sample_z, self)


def reparametrize_noise(x, gaussian_noise, vae_model):
    '''Transform gaussian noise into samples from VAE's variational approximation to the posterior

    e.g., draw eps ~ N(0, 1) and compute encoder_mu(x) + encoder_sigma(x) * eps

    Args:
        x: (Variable) observations; with shape n x m1
        gaussian_noise: (Variable) sampled noise; with shape n x m2
        vae_model: (VAE) the vector autoregressive model; implemented like a pytorch module

    Returns:
        (Variable) reparametrized gaussian noise s.t. imitate samples from q(z | x)
    '''
    mu = vae_model.encoder_mu(x)
    sigma = vae_model.encoder_sigma(x)

    reparametrized = torch.add(mu, torch.mul(sigma, gaussian_noise))

    return reparametrized


def _expand_batch_sigma_to_cov(sigma, m):
    '''Given a vector of scalar parameters sigma, expand into a batch of covariance matrices

    Assumes isotropic covariance, e.g., covariance = (sigma ** 2) * identity(m)

    Args:
        sigma: (Variable) batches of the scale parameter sigma; shape n x 1
        m: (int) the desired parameter

    Returns:
        (Variable) batches of the covariance matrices built from each sigma; shape n x m x m
    '''
    B, _ = sigma.size()

    sigma = sigma.unsqueeze(2)  # n x 1 x 1

    a = make_torch_variable(np.ones((m, 1)), requires_grad=False).expand(B, m, 1)
    b = make_torch_variable(np.ones((1, m)), requires_grad=False).expand(B, 1, m)
    c = make_torch_variable(np.identity(m), requires_grad=False).expand(B, m, m)

    # [(n x m x 1) x (n x 1 x 1)] x (n x 1 x m) element-wise (n x m x m) --> n x m x m
    cov = torch.bmm(torch.bmm(a, sigma ** 2), b) * c

    return cov


def vae_lower_bound(x, z, vae_model):
    '''Compute variational lower bound, as specified by variational autoencoder

    The VAE model specifies
      * likelihood x | z ~ MVN(encoder_mu(z), (encoder_sigma(z) ** 2) * I)
      * posterior z | x ~ MVN(decoder_mu(z), (decoder_sigma(z) ** 2) * I)
      * prior z ~ MVN(0, I)
    where the posterior is not the true posterior, but rather the variational approximation.

    This comes out to
      lower bound = log q(z | x) - log p(x, z)
                  = log q(z | x) - log p(x | z) - log p(z)

    Args:
        x: (Variable) observations; shape n x m1
        z: (Variable) latent variables; shape n x m2
        vae_model: (VAE) the vector autoregressive model; implemented like a pytorch module

    Returns:
        (Variable) lower bound; dim (1, )
    '''
    # Some initial parameter setting
    n, m1 = x.size()
    _, m2 = z.size()

    # Parameters of the likelihood of x given the model & z
    x_mu = vae_model.decoder_mu(z)  # b x m1
    x_sigma = _expand_batch_sigma_to_cov(vae_model.decoder_sigma(z), m1)  # b x m1 x m1

    # Parameters of the variational approximation of the posterior of z given model & x
    z_mu = vae_model.encoder_mu(x)  # b x m2
    z_sigma = _expand_batch_sigma_to_cov(vae_model.encoder_sigma(x), m2)  # b x m2 x m2

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
