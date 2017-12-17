#
# Latent variable models of form
#    z_i ~ MVN(0, I)
#    x_i ~ MVN(z_i.T * beta, sigma^2 * I)
# where z_i is R^K and x_i is R^M
#
# We assume all vectors are row-vectors; hence in the above, we have shapes:
#    z - (N, K)
#    x - (N, M)
#    beta - (K, M)
#    I - (M, M)
#
# We implement fitting with maximum likelihood, e-m, and variational bayes
#
from collections import namedtuple
import logging

import numpy as np
import torch
from torch.autograd import Variable

from code import mvn
from code import utils

logger = logging.getLogger(__name__)


def make_torch_variable(value, requires_grad, dtype=torch.FloatTensor):
    if not isinstance(value, torch.Tensor):
        value = torch.Tensor(value)
    return Variable(value.type(dtype), requires_grad=requires_grad)


def select_minibatch(x, B, replace=True):
    '''Shorthand for selecting B random elements & converting to autograd Variable'''
    # Remember, we assume that we're using row vectors!
    N, _ = x.shape
    ix_mini = np.random.choice(range(N), B, replace=replace)
    return make_torch_variable(x[ix_mini, :], False)


def gradient_descent_step(variable, learning_rate):
    if variable.grad is None:
        raise ValueError('no gradients')

    if variable.grad.data.numpy().sum() == 0:
        raise ValueError('gradients look zeroed out...')

    variable.data = variable.data - (learning_rate * variable.grad.data)

    return variable


def gradient_descent_step_parameter_tuple(parameter_tuple, learning_rate):
    for nm in parameter_tuple._fields:
        variable = parameter_tuple.__getattribute__(nm)
        logger.debug('Updating {name:s} (value, grad, step) = ({v:s}, {g:s}, {s:s}'
                     .format(name=nm, v=str(variable.data), g=str(variable.grad),
                             s=str(learning_rate)))
        gradient_descent_step(variable, learning_rate)


def clear_gradients(variable):
    if variable.grad is None:
        raise ValueError('no gradients')

    variable.grad.data.zero_()
    return variable


def clear_gradients_parameter_tuple(parameter_tuple):
    for nm in parameter_tuple._fields:
        variable = parameter_tuple.__getattribute__(nm)
        clear_gradients(variable)


# Methods for maximum likelihood model fitting

MLE_PARAMS = namedtuple('MLE_PARAMS', ['beta', 'sigma'])


def mle_initialize_parameters(M, K):
    '''Initialize the parameters before fitting'''
    beta = make_torch_variable(np.random.randn(K, M), True)
    sigma = make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    return MLE_PARAMS(beta=beta, sigma=sigma)


def _mle_expand_batch(variable, sub_B):
    '''Reshape (B, K) to (sub_B * B, K) such that (b + sub_b * B, k) = (b, k)'''
    B, K = variable.size()
    return variable.expand(sub_B, B, K).contiguous().view(sub_B * B, K)


def _mle_unpack_likelihood(variable, sub_B, B):
    '''Unpack likelihood from (N * M, ) to (N, M) s.t. (n + m * N, ) --> (n, m)'''
    utils.check_autograd_variable_size(variable, [(sub_B * B,)])
    return variable.view(sub_B, B)


def mle_estimate_batch_likelihood(batch, mle_params, sub_B, test_noise=None):
    # ### Validation

    # Check that params are of right type
    if not isinstance(mle_params, MLE_PARAMS):
        raise ValueError('parameter tuple must be of type MLE_PARAMS')

    # Check parameter sizes against batch size
    B, M = batch.size()
    K, M_1 = mle_params.beta.size()

    if M != M_1:
        raise AssertionError('batch and beta do not agree on M ({} vs {})'
                             .format((B, M), (K, M_1)))

    utils.check_autograd_variable_size(mle_params.sigma, [(1,)])

    # ### Computation

    # Sample noise required to compute monte carlo estimate of likelihood
    noise = make_torch_variable(np.random.randn(sub_B * B, K), False)
    if test_noise is not None:  # For debugging, allow insertion of a deterministic noise variable
        utils.check_autograd_variable_size(test_noise, [(sub_B * B, K)])
        noise = test_noise

    # Expand minibatch to match shape of noise
    batch = _mle_expand_batch(batch, sub_B)

    # Construct mu and sigma & compute density
    mu = torch.mm(noise, mle_params.beta)

    identity = make_torch_variable(np.identity(M), False)
    sigma = torch.mul(mle_params.sigma ** 2, identity)

    likelihood = mvn.torch_mvn_density(batch, mu, sigma)

    # Reshape density to (sub_B, B) and sum across first dimension
    utils.check_autograd_variable_size(likelihood, [(sub_B * B,)])
    likelihood = _mle_unpack_likelihood(likelihood, sub_B, B)

    # Compute approx expected likelihood of each batch sample
    approx_expected_likelihood_each_iter = likelihood.mean(dim=0)
    approx_marginal_log_likelihood = torch.log(approx_expected_likelihood_each_iter).sum()

    return approx_marginal_log_likelihood


def mle_forward_step(x, mle_params, B, sub_B, learning_rate):
    # Create minibatch
    batch = select_minibatch(x, B)

    # Estimate marginal likelihood of batch
    neg_marginal_log_lik = -1 * mle_estimate_batch_likelihood(batch, mle_params, sub_B)

    # Do a backward step
    neg_marginal_log_lik.backward()

    # Update step
    gradient_descent_step_parameter_tuple(mle_params, learning_rate)

    # Constrain sigma
    mle_params.sigma.data[0] = max(1e-10, mle_params.sigma.data[0])

    # Clear gradients
    clear_gradients_parameter_tuple(mle_params)

    return mle_params, neg_marginal_log_lik


def mle_forward_step_w_optim(x, mle_params, B, sub_B, optimizer):
    # Create minibatch
    batch = select_minibatch(x, B)

    # Estimate marginal likelihood of batch
    neg_marginal_log_lik = -1 * mle_estimate_batch_likelihood(batch, mle_params, sub_B)

    # Do a backward step
    neg_marginal_log_lik.backward()

    # Do a step
    optimizer.step()

    # Constrain sigma
    mle_params.sigma.data[0] = max(1e-10, mle_params.sigma.data[0])

    # Clear gradients
    optimizer.zero_grad()

    return mle_params, neg_marginal_log_lik


# Maximum likelihood - marginalizing out z

def compute_var(beta, sigma):
    '''Computes M = t(W) * W + sigma^2 * I, which is a commonly used quantity'''
    _, M = beta.size()
    identity = make_torch_variable(np.identity(M), False)
    a1 = torch.mm(beta.t(), beta)
    a2 = torch.mul(sigma ** 2, identity)
    return torch.add(a1, a2)


def mle_estimate_batch_likelihood_v2(batch, mle_params):
    if not isinstance(mle_params, MLE_PARAMS):
        raise ValueError('Input params must be of type MLE_PARAMS')

    B, M = batch.size()

    mu = make_torch_variable(np.zeros(M), False)
    sigma = compute_var(mle_params.beta, mle_params.sigma)

    approx_marginal_log_likelihood = mvn.torch_mvn_density(batch, mu, sigma, True)

    return approx_marginal_log_likelihood.sum()


def mle_forward_step_w_optim_v2(x, mle_params, B, optimizer):
    # Create minibatch
    batch = select_minibatch(x, B)

    # Estimate marginal likelihood of batch
    neg_marginal_log_lik = -1 * mle_estimate_batch_likelihood_v2(batch, mle_params)

    # Do a backward step
    neg_marginal_log_lik.backward()

    # Do a step
    optimizer.step()

    # Constrain sigma
    mle_params.sigma.data[0] = max(1e-10, mle_params.sigma.data[0])

    # Clear gradients
    optimizer.zero_grad()

    return mle_params, neg_marginal_log_lik


# Marginalize out beta and compute marginal likelihood

MLE_PARAMS_2 = namedtuple('MLE_PARAMS_2', ['z', 'sigma', 'alpha'])


def mle_initialize_parameters(N, M, K):
    '''Initialize the parameters before fitting'''
    z = make_torch_variable(np.random.randn(N, K), True)
    sigma = make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    alpha = make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    return MLE_PARAMS_2(z=z, sigma=sigma, alpha=alpha)


def mle_estimate_batch_likelihood_v3(x, batch_ix, mle_params_2):
    '''e.g., Y | x ~ N(0, alpha**2 t(x) x + sigma**2 I); treats each dimension independently'''
    B = batch_ix.shape[0]
    _, M = x.size()
    _, K = mle_params_2.z.size()

    batch_x = x[batch_ix, :]  # B x M
    batch_z = mle_params_2.z[batch_ix, :]  # B x K

    # ### Construct variance
    dot = torch.mm(batch_z, batch_z.t())  # (B x K) * (K x B) --> (B x B)
    identity = make_torch_variable(np.identity(B), False)
    var = torch.add(
        torch.mul(mle_params_2.alpha ** 2, dot),
        torch.mul(mle_params_2.sigma ** 2, identity)
    )

    # ### Compute log lik
    mu = make_torch_variable(np.zeros(B), requires_grad=False)
    approx_marginal_log_likelihood = mvn.torch_mvn_density(batch_x.t(), mu, var, log=True)

    return approx_marginal_log_likelihood


def mle_forward_step_w_optim_v3(x, mle_params, B, optimizer):
    # Create minibatch
    N, _ = x.shape
    batch_ix = np.random.choice(range(N), B, replace=True)

    # Estimate marginal likelihood of batch
    neg_marginal_log_lik = -1 * mle_estimate_batch_likelihood_v3(x, batch_ix, mle_params)

    # Do a backward step
    neg_marginal_log_lik.backward()

    # Do a step
    optimizer.step()

    # Constrain sigma
    mle_params.sigma.data[0] = max(1e-10, mle_params.sigma.data[0])

    # Clear gradients
    optimizer.zero_grad()

    return mle_params, neg_marginal_log_lik


# Expectation maximization

EM_PARAMS = namedtuple('EM_PARAMS', ['beta', 'sigma'])


def em_compute_posterior(batch, em_params):
    '''Computes the first and second moments of P(z | x)'''
    B, M = batch.size()
    K, _ = em_params.beta.size()

    var_inv = torch.inverse(compute_var(em_params.beta.t(), em_params.sigma))

    var_inv_ = var_inv.unsqueeze(0).expand(B, K, K) # K x K --> B x K x K
    beta_ = em_params.beta.unsqueeze(0).expand(B, K, M)  # K x M --> B x K x M
    batch_ = batch.unsqueeze(2)  # B x M --> B x M x 1

    # (B x K x K) * (B x K x M) * (B x M x 1) --> (B x K x 1)  --> (B x K)
    e_z = torch.bmm(var_inv_, torch.bmm(beta_, batch_)).squeeze(2)

    e_z_ = e_z.unsqueeze(2)  # (B x K) --> (B x K x 1)
    e_z_t_ = e_z.unsqueeze(1)  # (B x K) --> (B x 1 x K)

    a_1_ = torch.mul(em_params.sigma ** 2, var_inv_) # B x K x K
    a_2_ = torch.bmm(e_z_, e_z_t_)  # (K x 1) * (1 x K) --> B x K x K
    e_z2 = torch.add(a_1_, a_2_)

    return e_z, e_z2


def extract_diagonals(batch_matrices):
    '''Get diagonal out of a batch of square matrices (B x M x M) --> (B x M)'''
    B, I, I = batch_matrices.size()
    return torch.cat([batch_matrices[b, :, :].diag().unsqueeze(0) for b in range(B)], dim=0)


def em_compute_full_data_log_likelihood(batch, em_params, post_e_y, post_e_y2):
    B, M = batch.size()
    K, _ = em_params.beta.size()

    batch_ = batch.unsqueeze(2)  # B x M --> B x M x 1
    batch_t_ = batch.unsqueeze(1)  # B x M --> B x 1 x M
    beta_ = em_params.beta.unsqueeze(0).expand(B, K, M)  # K x M --> B x K x M
    beta_t_ = em_params.beta.t().unsqueeze(0).expand(B, K, M)  # K x M --> M x K --> B x M x K
    post_e_y_t_ = post_e_y.unsqueeze(1)  # B x K --> B x 1 x K

    # shape 1
    a_1 = M / 2.0 * torch.log(em_params.sigma ** 2)

    # shape B x 1 x1
    a_2 = 0.5 * extract_diagonals(post_e_y2).sum(dim=1)  # B x K x K --> B,
    a_2 = a_2.unsqueeze(1).unsqueeze(2)  # B --> B x 1 x 1

    # (B x 1 x M) * (B x M x 1) --> B x 1 x 1
    a_3_1 = 0.5 * (em_params.sigma ** -2)
    a_3_2 = torch.bmm(batch_t_, batch_)
    a_3 = torch.mul(a_3_1, a_3_2)

    # (B x 1 x K) * (B x K x M) * (B x M x 1) --> B x 1 x 1
    a_4_1 = -1 * (em_params.sigma ** -2)
    a_4_2 = torch.bmm(post_e_y_t_, torch.bmm(beta_, batch_))
    a_4 = torch.mul(a_4_1, a_4_2)

    # (B x K x M) * (B x M x K) * (B x K x K) -- (B x K x K) --> (B x 1)
    a_5_1 = 0.5 * (em_params.sigma ** -2)  # Dim 1,
    a_5_2 = torch.bmm(beta_, torch.bmm(beta_t_, post_e_y2))
    a_5 = torch.mul(a_5_1, extract_diagonals(a_5_2).sum(dim=1))
    a_5 = a_5.unsqueeze(1).unsqueeze(2)

    full_data_log_likelihood = (a_1 + a_2 + a_3 + a_4 + a_5).sum()

    return full_data_log_likelihood


def em_forward_step(x, em_params, B, optimizer):
    # Create minibatch
    batch = select_minibatch(x, B)

    # Compute posterior
    e_y, e_y2 = em_compute_posterior(batch, em_params)

    # Use posterior to construct full data log likelihood as function of em_params
    full_data_log_likelihood = em_compute_full_data_log_likelihood(batch, em_params, e_y, e_y2)

    # Do a backward step
    torch.mul(-1 * full_data_log_likelihood).backward()

    # Update step
    optimizer.step()

    # Constrain sigma
    em_params.sigma.data[0] = max(1e-10, em_params.sigma.data[0])

    # Clear gradients
    optimizer.zero_grad()

    return em_params, full_data_log_likelihood


# Methods for variational bayes fitting

VB_PARAMS = namedtuple('VB_PARAMS', ['beta', 'sigma', 'beta_q', 'sigma_q'])


def vb_initialize_parameters(M, K):
    beta = make_torch_variable(np.random.randn(K, M), True)
    sigma = make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    beta_q = make_torch_variable(np.random.randn(M, K), True)
    sigma_q = make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    return VB_PARAMS(beta=beta, sigma=sigma, beta_q=beta_q, sigma_q=sigma_q)


def _reparametrize_noise(batch, noise, vb_params):
    mu = torch.mm(batch, vb_params.beta_q)

    _, K = noise.size()
    identity = make_torch_variable(np.identity(K), False)
    sigma = torch.mul(vb_params.sigma_q ** 2, identity)

    return mu + torch.mm(noise, sigma)


def vb_estimate_lower_bound(batch, noise, vb_params):
    if not isinstance(vb_params, VB_PARAMS):
        raise ValueError('parameter tuple must be of type VB_PARAMS')

    B, M = batch.size()
    B_1, K = noise.size()

    if B != B_1:
        raise ValueError('Batch size is inconsistent between batch and noise')

    # Compute components
    mu_x = torch.mm(noise, vb_params.beta)
    identity_x = make_torch_variable(np.identity(M), False)
    sigma_x = torch.mul(vb_params.sigma ** 2, identity_x)

    mu_q = torch.mm(batch, vb_params.beta_q)
    identity_q = make_torch_variable(np.identity(K), False)
    sigma_q = torch.mul(vb_params.sigma_q ** 2, identity_q)

    mu_prior = make_torch_variable(np.zeros(K), False)
    sigma_prior = make_torch_variable(np.identity(K), False)

    # Compute log likelihoods
    log_posterior = mvn.torch_mvn_density(noise, mu_q, sigma_q, log=True)
    log_likelihood = mvn.torch_mvn_density(batch, mu_x, sigma_x, log=True)
    log_prior = mvn.torch_mvn_density(noise, mu_prior, sigma_prior, log=True)

    lower_bound = log_posterior - log_likelihood - log_prior

    return lower_bound.sum()


def vb_forward_step(x, vb_params, B, learning_rate):
    # Create minibatch
    batch = select_minibatch(x, B)

    # Sample noise
    K, _ = vb_params.beta.size()
    noise = make_torch_variable(np.random.randn(B, K), False)
    noise = _reparametrize_noise(batch, noise, vb_params)

    # Estimate marginal likelihood of batch
    neg_lower_bound = vb_estimate_lower_bound(batch, noise, vb_params)

    # Do a backward step
    neg_lower_bound.backward()

    # Update step
    gradient_descent_step_parameter_tuple(vb_params, learning_rate)

    # Constrain sigma
    vb_params.sigma.data[0] = max(1e-10, vb_params.sigma.data[0])
    vb_params.sigma_q.data[0] = max(1e-10, vb_params.sigma_q.data[0])

    # Clear gradients
    clear_gradients_parameter_tuple(vb_params)

    return vb_params, neg_lower_bound


def vb_forward_step_w_optim(x, vb_params, B, optimizer):
    # Create minibatch
    batch = select_minibatch(x, B)

    # Sample noise
    K, _ = vb_params.beta.size()
    noise = make_torch_variable(np.random.randn(B, K), False)
    noise = _reparametrize_noise(batch, noise, vb_params)

    # Estimate marginal likelihood of batch
    neg_lower_bound = vb_estimate_lower_bound(batch, noise, vb_params)

    # Do a backward step
    neg_lower_bound.backward()

    # Update step
    optimizer.step()

    # Constrain sigma
    vb_params.sigma.data[0] = max(1e-10, vb_params.sigma.data[0])
    vb_params.sigma_q.data[0] = max(1e-10, vb_params.sigma_q.data[0])

    # Clear gradients
    optimizer.zero_grad()

    return vb_params, neg_lower_bound
