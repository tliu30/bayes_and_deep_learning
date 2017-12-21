#
# Latent variable models of form
#    z_i ~ MVN(0, I)
#      f ~ GP(0, K(z_1, z_2))
#    x_i ~ MVN(f(z_i), sigma^2 * I)
# where z_i is R^M2 and x_i is R^M1
#
# I think there's an assumption here of x being centered & isotropic
#
# We assume all vectors are row-vectors; hence in the above, we have shapes:
#    z - (N, M2)
#    x - (N, M1)
#
# We implement two methods:
#    (1) fitting with maximum likelihood w.r.t. z (e.g., marginalize out f)
#    (2) variational inference with q(z_i | x_i) = N(g(x_i), tau^2 ** I)
#        where q ~ GP(0, K'(x_1, x_2))
#
from collections import namedtuple

import numpy as np
import torch

from code.mvn import torch_mvn_density
from code.utils import make_torch_variable


# Kernel functions

def rbf_kernel_forward(x1, x2, log_lengthscale, eps=1e-5):
    '''Compute the distance matrix under rbf kernel; taken from github repo jrg365/gpytorch

    e.g., K(x, x') = exp( - ||x - x'||^2 / (l + eps) )

    Args:
        x1: (torch variable) shape n x d
        x2: (torch variable) shape m x d
        log_lengthscale: (torch variable) log of the hyperparameter l
        eps: (float) a safety term to make the lengthscale not zero (since is in denominator)

    Returns:
        distance matrix of shape (n x m)
    '''
    n, d = x1.size()
    m, _ = x2.size()

    res = 2 * x1.matmul(x2.transpose(0, 1))

    x1_squared = torch.bmm(x1.view(n, 1, d), x1.view(n, d, 1))
    x1_squared = x1_squared.view(n, 1).expand(n, m)
    x2_squared = torch.bmm(x2.view(m, 1, d), x2.view(m, d, 1))
    x2_squared = x2_squared.view(1, m).expand(n, m)
    res.sub_(x1_squared).sub_(x2_squared)  # res = -(x - z)^2

    res = res / (log_lengthscale.exp() + eps)  # res = -(x - z)^2 / lengthscale
    res.exp_()

    return res


# Maximum likelihood w.r.t. z

MLE_PARAMS = namedtuple('MLE_PARAMS', ['z', 'alpha', 'sigma', 'log_l'])


def _make_cov(x1, x2, alpha, sigma, log_l):
    '''Construct the covariance matrix for observations drawn from a gaussian process

    In particular, we compute K(x1, x2) + sigma ** 2, e.g., the covariance between each point
    in x1 with each point in x2 given the parameters.

    where k(x1, x2) = (alpha ** 2) * exp(-0.5 * l * (t(x1) * x2)) (e.g., assuming RBF kernel)

    Args:
        x1: (Variable) the first set of points; shape N1 x M
        x2: (Variable) the second set of points; shape N2 x M
        alpha: (Variable) scale parameter for the RBF
        sigma: (Variable) observation noise
        log_l: (Variable) log length parameter for RBF

    Returns:
        (Variable) with shape N1 x N2
    '''
    n1, _ = x1.size()
    n2, _ = x2.size()

    inner_product = rbf_kernel_forward(x1, x2, log_l)
    identity = make_torch_variable(np.identity(n1)[:n1, :n2], False)

    a1 = torch.mul(alpha ** 2, inner_product)
    a2 = torch.mul(sigma ** 2, identity)
    cov = torch.add(a1, a2)

    return cov


def _mle_log_likelihood(x, z, alpha, sigma, log_l):
    '''Compute the log likelihood P(X | Z, alpha, sigma, log_l)

    In particular, if X is (N x M1) and Z is (N x M2), we compute

        P(X | Z, alpha, sigma, log_l) = prod_j P(X_{-, j} | Z, alpha, sigma, log_l)
                                      = prod_j MVN(X_{-, j}; 0, K)

    e.g., assuming independence between columns of X, and where K is a (N x N) matrix with entries

        K[i, j] = (alpha ** 2) * rbf(z_i, z_j) + (sigma ** 2) * I

    Note that we presume the same covariance structure across each dimension in this simple
    model...

    Args:
        x: (Variable) N x M1 matrix of observations
        z: (Variable) N x M2 matrix of latent variables
        alpha: (Variable) the scale parameter of the gp kernel (maybe a redundant hyperparameter...)
        sigma: (Variable) observation noise
        log_l: (Variable) log lengthscale parameter of RBF

    Returns:
        (Variable) the marginal log likelihood of the observations X as a (1, ) vector
    '''
    n, m1 = x.size()
    _, m2 = z.size()

    # Compute covariance matrix of each dimension (shape n x n)
    cov = _make_cov(z, z, alpha, sigma, log_l)

    # Compute log lik
    mu = make_torch_variable(np.zeros(n), requires_grad=False)
    approx_marginal_log_likelihood = torch_mvn_density(x.t(), mu, cov, log=True).sum()

    return approx_marginal_log_likelihood


def mle_batch_log_likelihood(x, mle_params, batch_ix):
    '''Compute log likelihood given latent variables and parameters'''
    batch_x = x[batch_ix, :]
    batch_z = mle_params.z[batch_ix, :]
    return _mle_log_likelihood(batch_x, batch_z, mle_params.alpha, mle_params.sigma, mle_params.log_l)


def mle_forward_step_w_optim(x, mle_params, b, optimizer):
    '''This is naive minibatch optimiziation of MLE...'''
    # Create batch
    n, _ = x.shape
    batch_ix = np.random.choice(range(n), b, replace=True)

    # Estimate marginal likelihood
    neg_log_lik = -1 * mle_batch_log_likelihood(x, mle_params, batch_ix)

    # Do a backward step
    neg_log_lik.backward()

    # Do an optimization step
    optimizer.step()

    # Enforce bounds on lengthscale and sigmas
    mle_params.log_l.data[0] = max(-10000, mle_params.log_l.data[0])
    mle_params.log_l.data[0] = min(10000, mle_params.log_l.data[0])
    mle_params.sigma.data[0] = max(1e-10, mle_params.sigma.data[0])
    mle_params.alpha.data[0] = max(1e-10, mle_params.alpha.data[0])

    # Clear gradients
    optimizer.zero_grad()

    return mle_params, neg_log_lik


# Implement Lawrence GPLVM active / inactive algorithm


def _gp_conditional_mean_cov(cur_z, active_x, active_z, alpha, sigma, log_l,
                             active_cov=None, active_cov_inv=None, mean_component=None):
    '''Compute E[x] and Var(x) where x ~ N(f(cur_z), sigma) and f | active ~ GP(0, alpha, l)

    In other words, compute the mean and variance of x given it's latent point z conditioned
    on the active set.
    '''
    # Extract dimensions
    active_n, m1 = active_x.size()
    cur_n, _ = cur_z.size()

    if cur_n > 1:
        raise ValueError('At the moment we just do one inactive point at a time...')

    # Construct covariance of active set & functions thereof that get reused
    precomputed = not ((active_cov is None) | (active_cov_inv is None) | (mean_component is None))
    if not precomputed:
        active_cov = _make_cov(active_z, active_z, alpha, sigma, log_l)
        active_cov_inv = torch.inverse(active_cov)
        mean_component = torch.mm(active_x.t(), active_cov_inv)  # Dim (M1 x active_n)

    # Compute covariance of inactive point with active set, as well as with itself
    no_noise = make_torch_variable([0.0], requires_grad=False)
    active_inactive_cov = _make_cov(active_z, cur_z, alpha, no_noise, log_l)  # Dim (active_n x 1)
    inactive_var = _make_cov(cur_z, cur_z, alpha, sigma, log_l)  # Dim (1 x 1)

    # Compute E[cur_x] = t(active_x) * inv(active(cov)) * active_inactive_cov
    # Dim (m1 x active_n) * (active_n x active_n) * (active_n x 1) --> (m1 x 1)
    inactive_mu = torch.mm(mean_component, active_inactive_cov)

    # Compute Cov(cur_x) = (sigma ** 2) * I
    # Dim of sigma (1 x 1) - (1 x active_n) * (active_n x active_n) * (active_n x 1) --> (1 x 1)
    a1 = torch.mm(active_inactive_cov.t(), torch.mm(active_cov_inv, active_inactive_cov))
    identity = make_torch_variable(np.identity(m1), False)
    inactive_sigma = torch.mul(torch.add(inactive_var, -1 * a1), identity)

    return inactive_mu, inactive_sigma


def _inactive_point_likelihood(active_x, active_z, inactive_x, inactive_z, alpha, sigma, log_l):
    '''Compute the likelihood of the inactive set conditional on the active set'''
    # Extract dimensions
    active_n, m1 = active_x.size()
    inactive_n, _ = inactive_x.size()

    # Construct covariance of active set & functions thereof that get reused
    active_cov = _make_cov(active_z, active_z, alpha, sigma, log_l)
    active_cov_inv = torch.inverse(active_cov)
    mean_component = torch.mm(active_x.t(), active_cov_inv)  # Dim (M1 x active_n)

    # Compute likelihood of inactive set conditional on active set
    # (we iterate since (1) all computations are independent and (2) for clarity)
    inactive_log_lik = make_torch_variable([0], requires_grad=False)
    for i in range(inactive_n):

        # Extract current pair
        cur_x = inactive_x[[i], :]
        cur_z = inactive_z[[i], :]

        # Compute log likelihood of current instance
        inactive_mu, inactive_sigma = _gp_conditional_mean_cov(
            cur_z, active_x, active_z, alpha, sigma, log_l,
            active_cov=active_cov, active_cov_inv=active_cov_inv, mean_component=mean_component
        )
        instance_log_lik = torch_mvn_density(cur_x, inactive_mu, inactive_sigma, log=True)

        # Add to overall log likelihood
        inactive_log_lik = torch.add(inactive_log_lik, instance_log_lik)

    return inactive_log_lik


def inactive_point_likelihood(x, mle_params, active_ix, inactive_ix):
    '''Wrapper around the other function'''
    active_x = x[active_ix, :]
    active_z = mle_params.z[active_ix, :]

    inactive_x = x[inactive_ix, :]
    inactive_z = mle_params.z[inactive_ix, :]

    return _inactive_point_likelihood(
        active_x, active_z, inactive_x, inactive_z,
        mle_params.alpha, mle_params.sigma, mle_params.log_l
    )


def mle_active_inactive_step_w_optim(x, mle_params, b, optimizer_kernel, optimizer_latent):
    '''From GPLVMs for Visualiation of High Dimensional Data by Neil Lawrence


    1. Split data in to active & inactive set
    2. Restricting to active set, learn parameters of the kernel
    3. Using active set and fixing kernel parameters, estimate latent variables of inactive set
    4. Rinse and repeat
    '''
    # Create batch (they use informative vector machine [IVM] but i'm lazy)
    n, _ = x.shape
    active_ix = np.random.choice(range(n), b, replace=False)  # w/o replacement!
    inactive_ix = [x for x in range(n) if x not in active_ix]

    # Optimize kernel parameters given the active set
    # TODO: refactor mle_forward_step_w_optim to do this?
    neg_active_log_lik = -1 * mle_batch_log_likelihood(x, mle_params, active_ix)

    neg_active_log_lik.backward()
    optimizer_kernel.step()

    mle_params.log_l.data[0] = max(-10000, mle_params.log_l.data[0])
    mle_params.log_l.data[0] = min(10000, mle_params.log_l.data[0])
    mle_params.sigma.data[0] = max(1e-10, mle_params.sigma.data[0])
    mle_params.alpha.data[0] = max(1e-10, mle_params.alpha.data[0])

    # Optimize latent parameters of inactive set
    # TODO: do i need to do some splitting here so it doesn't try to optimize active latents?
    neg_inactive_log_lik = -1 * inactive_point_likelihood(x, mle_params, active_ix, inactive_ix)

    neg_inactive_log_lik.backward()
    optimizer_latent.step()

    return mle_params, neg_active_log_lik, neg_inactive_log_lik


# Variational bayes version - where q(z) ~ N(g(x), sigma), g ~ GP

VB_PARAMS = namedtuple(
    'VB_PARAMS',
    ['alpha', 'sigma', 'log_l', 'alpha_q', 'sigma_q', 'log_l_q']
)


def _vb_lower_bound(x, noise, alpha, sigma, log_l, alpha_q, sigma_q, log_l_q):
    '''Compute variational lower bound'''

    B, M1 = x.size()
    _, M2 = noise.size()

    # Posterior under variational approximation
    mu_z = make_torch_variable(np.zeros(B), requires_grad=False)
    cov_z = _make_cov(x, x, alpha_q, sigma_q, log_l_q)  # for f_q: X --> Z
    log_posterior = torch_mvn_density(noise.t(), mu_z, cov_z)

    # Likelihood under model
    mu_x = make_torch_variable(np.zeros(B), requires_grad=False)
    cov_x = _make_cov(noise, noise, alpha, sigma, log_l)  # for f: Z --> X
    log_likelihood = torch_mvn_density(x.t(), mu_x, cov_x)

    # Prior under model
    mu_prior = make_torch_variable(np.zeros(M2), requires_grad=False)
    sigma_prior = make_torch_variable(np.identity(M2), requires_grad=False)
    log_prior = torch_mvn_density(noise, mu_prior, sigma_prior, log=True)

    # Compute lower bound
    lower_bound = log_posterior - log_likelihood - log_prior

    return lower_bound.sum()


def vb_lower_bound(x, noise, vb_params):
    '''Compute variational lower bound'''
    return _vb_lower_bound(
        x, noise,
        vb_params.alpha, vb_params.sigma, vb_params.log_l,
        vb_params.alpha_q, vb_params.sigma_q, vb_params.log_l_q
    )


def _reparametrize_noise(inactive_x, inactive_noise, active_x, active_z, alpha_q, sigma_q, log_l_q):  # Don't i need to choose an active set?
    '''Using an active set to give some definition to GP draw, reparam N(0, 1) noise given batch'''
    # Extract dimensions
    active_n, m1 = active_x.size()
    inactive_n, _ = inactive_x.size()

    # Construct covariance of active set & functions thereof that get reused
    active_cov = _make_cov(active_x, active_x, alpha_q, sigma_q, log_l_q)
    active_cov_inv = torch.inverse(active_cov)
    mean_component = torch.mm(active_z.t(), active_cov_inv)  # Dim (M1 x active_n)

    # Compute likelihood of inactive set conditional on active set
    # (we iterate since (1) all computations are independent and (2) for clarity)
    reparametrized_points = []
    for i in range(inactive_n):

        # Extract current pair
        cur_x = inactive_x[[i], :]
        cur_noise = inactive_noise[[i], :]

        inactive_mu, inactive_sigma = _gp_conditional_mean_cov(
            cur_x, active_z, active_x, alpha_q, sigma_q, log_l_q,
            active_cov=active_cov, active_cov_inv=active_cov_inv, mean_component=mean_component
        )

        reparametrized_points.append(torch.add(inactive_mu, torch.mul(inactive_sigma, cur_noise)))

    return torch.cat(reparametrized_points, dim=1)  # The dimension might be wrong here


def reparametrize_noise(inactive_x, inactive_noise, active_x, active_z, vb_params):
    '''Using an active set to give some definition to GP draw, reparam N(0, 1) noise given batch'''
    return _reparametrize_noise(
        inactive_x, inactive_noise, active_x, active_z,
        vb_params.alpha_q, vb_params.sigma_q, vb_params.log_l_q
    )


def resample_latent_space(batch_x, active_x, active_z, vb_params):
    '''e.g., resample latent points z corresponding to batch variables given an active set'''
    n_batch, _ = batch_x.size()
    _, m2 = active_z.size()

    # Do this to get the mean...alternatively, could sample to introduce noise
    noise = make_torch_variable(np.zeros((n_batch, m2)), requires_grad=False)

    return reparametrize_noise(batch_x, noise, active_x, active_z, vb_params)


def vb_forward_step(x, z, vb_params, n_active, n_batch, optimizer):
    '''The iterative step for a gaussian process latent variable model with variational inference

    Our goal here was to imitate the setup used by D.P. Kingma for AEVB, e.g.,
        1. Define the likelihood as x ~ N(f(z), sigma) where f: Z -> X ~ P(.)
        2. Define the variational approximation as part of the same family of distributions as the
           likelihood, e.g., z ~ q(z | x) = N(f_q(x), sigma_q) where g: X -> Z ~ P_q(.)

    However, it's a bit odd for gaussian processes because unlike AEVB - where P & P_q are
    parametrized by the weights of the neural network, and therefore which establish a well-defined
    mean f(z) & f_q(x) - the gaussian process framework often has a mean of zero unless we condition
    it on an active set...

    For that reason, we try to hack an approach together where we establish an active set before
    computing the variational lower bound, as it creates a more meaningful function...
    '''
    # Select an active set
    n, m1 = x.size()
    _, m2 = z.size()

    active_ix = np.random.choice(x, size=n_active, replace=False)

    active_x = x[active_ix, :]
    active_z = z[active_ix, :]

    # Select a batch from inactive set
    inactive_ix = np.array([x for x in range(n) if x not in active_ix])
    batch_ix = np.random.choice(inactive_ix, size=n_batch, replace=True)

    batch_x = x[batch_ix, :]

    # Sample noise for batch (sample as MVN(0, I))
    noise = np.random.randn(n, m2)

    # Reparametrize noise given an active set
    batch_z = reparametrize_noise(batch_x, noise, active_x, active_z, vb_params)

    # Compute variational lower bound & update parameters
    lower_bound = vb_lower_bound(batch_x, batch_z, vb_params)

    lower_bound.backward()
    optimizer.step()

    # Constrain sigma & alpha
    vb_params.sigma.data[0] = max(1e-10, vb_params.sigma.data[0])
    vb_params.alpha.data[0] = max(1e-10, vb_params.alpha.data[0])
    vb_params.sigma_q.data[0] = max(1e-10, vb_params.sigma_q.data[0])
    vb_params.alpha_q.data[0] = max(1e-10, vb_params.alpha_q.data[0])

    # Clear gradients
    optimizer.zero_grad()

    # Update latent variables for inactive set given new parameters
    z[batch_ix, :] = resample_latent_space(batch_x, active_x, active_z, vb_params)
