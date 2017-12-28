import numpy as np
import torch
from torch import nn

from code.mvn import torch_mvn_density, torch_diagonal_mvn_density_batch
from code.utils import make_torch_variable, select_minibatch


class FullyConnectedNN(nn.Module):

    def __init__(self, layer_dimensions):
        super(FullyConnectedNN, self).__init__()
        layer_pairs = zip(layer_dimensions[:-1], layer_dimensions[1:])
        self.layers = nn.ModuleList([nn.Linear(n_in, n_out) for n_in, n_out in layer_pairs])

        self.input_dim = layer_dimensions[0]
        self.output_dim = layer_dimensions[-1]

    def forward(self, x):
        input = x
        for i in range(len(self.layers) - 1):
            input = self.layers[i](input).clamp(min=0)
        out = self.layers[-1](input)
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

        # return vae_lower_bound(x, sample_z, self)
        return vae_lower_bound_less_sampling(x, sample_z, self)


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
    x_mu = vae_model.decoder_mu(z)  # (b, m1)
    x_sigma = vae_model.decoder_sigma(z).squeeze(1)  # (b, )

    # Parameters of the variational approximation of the posterior of z given model & x
    z_mu = vae_model.encoder_mu(x)  # (b, m2)
    z_sigma = vae_model.encoder_sigma(x).squeeze(1)  # (b, )

    # Parameters of the prior of z
    prior_mu = make_torch_variable(np.zeros(m2), requires_grad=False)  # dim: m2
    prior_sigma = make_torch_variable(np.identity(m2), requires_grad=False)  # m2 x m2

    # Compute components
    log_posterior = torch_diagonal_mvn_density_batch(z, z_mu, z_sigma, log=True)
    log_likelihood = torch_diagonal_mvn_density_batch(x, x_mu, x_sigma, log=True)
    log_prior = torch_mvn_density(z, prior_mu, prior_sigma, log=True)

    lower_bound = (log_posterior - log_likelihood - log_prior).sum()

    return lower_bound


def vae_lower_bound_less_sampling(x, z, vae_model):
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
    x_mu = vae_model.decoder_mu(z)  # (b, m1)
    x_sigma = vae_model.decoder_sigma(z).squeeze(1)  # (b, )

    # Parameters of the variational approximation of the posterior of z given model & x
    z_mu = vae_model.encoder_mu(x)  # (b, m2)
    z_sigma = vae_model.encoder_sigma(x).squeeze(1)  # (b, )

    # Compute components (e.g., expected log ___ under posterior approximation)
    log_posterior = 0.5 * m2 * (1 + torch.log(z_sigma ** 2)) + 0.5 * m2 * np.log(2 * np.pi)
    log_likelihood = torch_diagonal_mvn_density_batch(x, x_mu, x_sigma, log=True)
    log_prior = 0.5 * ((z_mu ** 2).sum() + (z_sigma ** 2)) + 0.5 * m2 * np.log(2 * np.pi)

    lower_bound = -1 * (log_posterior - log_likelihood - log_prior).sum()

    return lower_bound


def vae_forward_step_w_optim(x, model, B, optimizer):
    # Clear gradients
    optimizer.zero_grad()

    # Create minibatch
    batch = select_minibatch(x, B, replace=False)

    # Evaluate loss
    lower_bound = model(batch)

    # Backward step
    lower_bound.backward()

    # Update step
    optimizer.step()

    return model, lower_bound
