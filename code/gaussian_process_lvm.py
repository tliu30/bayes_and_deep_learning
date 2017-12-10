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
from code.mvn import torch_determinant, torch_gp_mvn_log_density, torch_mvn_density


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


def _make_cov(x1, x2, alpha, sigma, log_l):
    n1, _ = x1.size()
    n2, _ = x2.size()

    inner_product = rbf_kernel_forward(x1, x2, log_l)
    identity = make_torch_variable(np.identity(n1)[:n1, :n2], False)

    a1 = torch.mul(alpha ** 2, inner_product)
    a2 = torch.mul(sigma ** 2, identity)
    cov = torch.add(a1, a2)

    return cov


def _mle_log_likelihood(x, z, alpha, sigma, log_l):
    '''Note we use just one lengthscale across every dimension'''
    n, m1 = x.size()
    _, m2 = z.size()

    # Compute covariance matrix of each dimension (shape n x n)
    cov = _make_cov(z, z, alpha, sigma, log_l)

    # Compute log lik
    approx_marginal_log_likelihood = torch_gp_mvn_log_density(x, cov)

    return approx_marginal_log_likelihood


def mle_batch_log_likelihood(x, mle_params, batch_ix):
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


def _inactive_step(x, mle_params, active_ix, inactive_ix):
    _, m1 = x.size()

    active_x = x[active_ix, :]
    active_z = mle_params.z[active_ix, :]
    active_n, _ = active_x.size()

    active_cov = _make_cov(active_z, active_z, mle_params.alpha, mle_params.sigma, mle_params.log_l)
    active_cov_inv = torch.inverse(active_cov)

    inactive_log_lik = make_torch_variable([0], requires_grad=False)
    for i in inactive_ix:
        inactive_x = x[[i], :]
        inactive_z = mle_params.z[[i], :]

        active_inactive_cov = _make_cov(active_z, inactive_z, mle_params.alpha, mle_params.sigma,
                                        mle_params.log_l)
        inactive_var = _make_cov(inactive_z, inactive_z, mle_params.alpha, mle_params.sigma,
                                 mle_params.log_l)

        # TODO: Do not have dimensions right here...
        # (M1 x 1) x (active_n x active_n) x (active_n x 1) --> (M1 x 1)
        inactive_mu = torch.mm(inactive_x.t(), torch.mm(active_cov_inv, active_inactive_cov))

        # (1 x active_n) x (active_n x active_n) x (active_n x 1) --> (1 x 1)
        a1 = torch.mm(active_inactive_cov.t(), torch.mm(active_cov_inv, active_inactive_cov))
        identity = make_torch_variable(np.identity(m1), False)
        inactive_sigma = torch.mul(torch.add(inactive_var, -1 * a1), identity)

        # Compute log likelihood of current instance
        instance_log_lik = torch_mvn_density(inactive_x, inactive_mu, inactive_sigma, log=True)

        inactive_log_lik = torch.add(inactive_log_lik, instance_log_lik)

    return inactive_log_lik


def mle_active_inactive_step_w_optim(x, mle_params, b, optimizer_kernel, optimizer_latent):
    '''From GPLVMs for Visualiation of High Dimensional Data by Neil Lawrence'''
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
    neg_inactive_log_lik = -1 * _inactive_step(x, mle_params, active_ix, inactive_ix)

    neg_inactive_log_lik.backward()
    optimizer_latent.step()
