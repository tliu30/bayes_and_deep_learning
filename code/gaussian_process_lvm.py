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

# Kernel functions
from code.linear_regression_lvm import make_torch_variable
from code.mvn import torch_determinant, torch_gp_mvn_log_density


def rbf_kernel_forward(x1, x2, log_lengthscale, eps=1e-5):
    '''Compute the distance matrix under rbf kernel; taken from github repo jrg365/gpytorch

    e.g., K(x, x') = exp( - ||x - x'||^2 / (2 * sigma**2) )

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


def _mle_likelihood(x, z, alpha, sigma, log_l, eps=1e-5):
    '''Note we use just one lengthscale across every dimension'''
    n, m1 = x.size()
    _, m2 = z.size()

    # Compute covariance matrix of each dimension (shape n x n)
    inner_product = rbf_kernel_forward(z, z, log_l, eps=eps)
    identity = make_torch_variable(np.identity(n), True)
    cov = torch.add(torch.mul(alpha ** 2, inner_product), torch.mul(sigma ** 2, identity))

    # Compute log lik
    approx_marginal_log_likelihood = torch_gp_mvn_log_density(x, cov)

    return approx_marginal_log_likelihood


def mle_batch_likelihood(x, mle_params, batch_ix):
    batch_x = x[batch_ix, :]
    batch_z = mle_params.z[batch_ix, :]
    return _mle_likelihood(batch_x, batch_z, mle_params.alpha, mle_params.sigma, mle_params.log_l)


def mle_forward_step_w_optim(x, mle_params, b, optimizer):
    '''This is naive minibatch optimiziation of MLE...'''
    # Create batch
    n, _ = x.shape
    batch_ix = np.random.choice(range(n), b, replace=True)

    # Estimate marginal likelihood
    neg_log_lik = -1 * mle_batch_likelihood(x, mle_params, batch_ix)

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


def _active_step(x, mle_params, active_ix, optimizer_kernel):
    active_x = x[active_ix, :]
    active_z = mle_params.z[active_ix, :]




def mle_active_inactive_step_w_optim(x, mle_params, b, optimizer_kernel, optimizer_latent):
    '''From GPLVMs for Visualiation of High Dimensional Data by Neil Lawrence'''
    # Create batch (they use informative vector machine [IVM] but i'm lazy)
    n, _ = x.shape
    active_ix = np.random.choice(range(n), b, replace=False)  # w/o replacement!

    # Optimize kernel parameters given the active set



