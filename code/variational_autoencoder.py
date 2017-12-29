import numpy as np
import torch
from torch import nn

from code.mvn import torch_diagonal_mvn_density_batch
from code.utils import make_torch_variable, select_minibatch


class VAE(nn.Module):

    def __init__(self, m1, m2, n_hidden):
        '''Construct a variational autoencoder by providing encoder and decoder networks

        Remember that "encoders" parametrize the variational approximation to the posterior
            q(z | x) = MVN(encoder_mu(x), (encoder_logvar(x) ** 2) * I)
        and the "decoders" parametrize the likelihood
            p(x | z) = MVN(decoder_mu(z), (decoder_logvar(z) ** 2) * I)

        Args:
            m: (int) dimension of observations
            k: (int) dimension of latent variables
            encoder_mu: (nn.Module) n x m--> n x k; specifies posterior's location parameter
            encoder_logvar: (nn.Module) n x m--> n x k; specifies posterior's scale parameter
            decoder_mu: (nn.Module) n x k --> n x m; specifies likelihood's location parameter
            decoder_logvar: (nn.Module) n x k--> n x m; specifies likelihood's scale parameter
        '''
        super(VAE, self).__init__()

        self.encoder_hidden = nn.Linear(m1, n_hidden)
        self.encoder_mu_out = nn.Linear(n_hidden, m2)
        self.encoder_logvar_out = nn.Linear(n_hidden, m2)

        self.decoder_hidden = nn.Linear(m2, n_hidden)
        self.decoder_mu_out = nn.Linear(n_hidden, m1)
        self.decoder_logvar_out = nn.Linear(n_hidden, m1)

        self.m1 = m1
        self.m2 = m2

    def encode(self, x):
        h = nn.ReLU(self.encoder_hidden(x))
        return self.encoder_mu_out(h), self.encoder_logvar_out(h)

    def decode(self, z):
        h = nn.ReLU(self.decoder_hidden(z))
        return self.decoder_mu_out(h), self.decoder_logvar_out(h)

    def forward(self, x, noise=None):
        '''Estimate variational lower bound of parameters given observations x'''
        n, _ = x.size()

        if noise is None:
            noise = make_torch_variable(np.random.randn(n, self.k), requires_grad=False)

        sample_z = reparametrize_noise(x, noise, self)

        # return vae_lower_bound(x, sample_z, self)
        return vae_lower_bound(x, sample_z, self)


class VAETest(VAE):

    def __init__(self, m1, m2, encoder_mu, encoder_logvar, decoder_mu, decoder_logvar):
        self.encoder_mu = encoder_mu
        self.encoder_logvar = encoder_logvar
        self.decoder_mu = decoder_mu
        self.decoder_logvar = decoder_logvar
        self.m1 = m1
        self.m2 = m2

    def encode(self, x):
        return self.encoder_mu(x), self.encoder_logvar(x)

    def decode(self, z):
        return self.decoder_mu(z), self.decoder_logvar(z)


def reparametrize_noise(x, gaussian_noise, vae_model):
    '''Transform gaussian noise into samples from VAE's variational approximation to the posterior

    e.g., draw eps ~ N(0, 1) and compute encoder_mu(x) + np.exp(2 * encoder_logvar(x)) * eps

    Args:
        x: (Variable) observations; with shape n x m1
        gaussian_noise: (Variable) sampled noise; with shape n x m2
        vae_model: (VAE) the vector autoregressive model; implemented like a pytorch module

    Returns:
        (Variable) reparametrized gaussian noise s.t. imitate samples from q(z | x)
    '''
    mu, logvar = vae_model.encode(x)
    sigma = torch.exp(torch.mul(logvar, 0.5))

    reparametrized = torch.add(mu, torch.mul(sigma, gaussian_noise))

    return reparametrized


def vae_lower_bound(x, z, vae_model):
    '''Compute variational lower bound, as specified by variational autoencoder

    The VAE model specifies
      * likelihood x | z ~ MVN(encoder_mu(z), (encoder_logvar(z) ** 2) * I)
      * posterior z | x ~ MVN(decoder_mu(z), (decoder_logvar(z) ** 2) * I)
      * prior z ~ MVN(0, I)
    where the posterior is not the true posterior, but rather the variational approximation.

    This comes out to
      lower bound = E[log q(z | x) - log p(x, z)]
                  = E[log q(z | x) - log p(x | z) - log p(z)]
    where the expectation is over z ~ q(z | x).

    Args:
        x: (Variable) observations; shape n x m1
        z: (Variable) latent variables; shape n x m2
        vae_model: (VAE) the vector autoregressive model; implemented like a pytorch module

    Returns:
        (Variable) lower bound; dim (1, )
    '''
    # ### Get parameters

    # Some initial parameter setting
    n, m1 = x.size()
    _, m2 = z.size()

    # Parameters of the likelihood of x given the model & z
    x_mu, _ = vae_model.decode(z)
    x_sigma = make_torch_variable(torch.ones(n), requires_grad=False)

    # Parameters of the variational approximation of the posterior of z given model & x
    z_mu, z_logvar = vae_model.encode(x)

    # ### Compute components (e.g., expected log ___ under posterior approximation)

    # E[log posterior]: analytically, is -0.5 * sum_j [log(2 pi) + 1 + log(sigma_j ** 2)]
    log_posterior = -0.5 * (z_logvar + 1 + np.log(2 * np.pi)).sum(dim=1)  # (B, )

    # E[log likelihood]: can't get analytically, so we use the sample estimate...
    log_likelihood = torch_diagonal_mvn_density_batch(x, x_mu, x_sigma, log=True)  # (B, )

    # E[log prior]: analytically, is -0.5 * sum_j [log(2 pi) + mu_j ** 2 + sigma_j ** 2]
    log_prior = -0.5 * ((z_mu ** 2) + torch.exp(z_logvar) + np.log(2 * np.pi)).sum(dim=1)  # (B, )

    # ### Put it all together
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
