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

import numpy as np
import torch

from code import mvn
from code import utils


# ### Maximum likelihood estimation methods


# Utilities

MLE_PARAMS = namedtuple('MLE_PARAMS', ['beta', 'sigma'])


def mle_initialize_parameters(M, K):
    '''Initialize the parameters before fitting'''
    beta = utils.make_torch_variable(np.random.randn(K, M), True)
    sigma = utils.make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    return MLE_PARAMS(beta=beta, sigma=sigma)


def _mle_expand_batch(variable, sub_B):
    '''Reshape (B, K) to (sub_B * B, K) such that (b + sub_b * B, k) = (b, k)'''
    B, K = variable.size()
    return variable.expand(sub_B, B, K).contiguous().view(sub_B * B, K)


def _mle_unpack_likelihood(variable, sub_B, B):
    '''Unpack likelihood from (N * M, ) to (N, M) s.t. (n + m * N, ) --> (n, m)'''
    utils.check_autograd_variable_size(variable, [(sub_B * B,)])
    return variable.view(sub_B, B)


#
# Method 1:
#
# A naive approach - we directly try to estimate beta & sigma by maximizing
#
#    log P(x | beta, sigma^2) = sum_i log P(x_i | beta, sigma)
#                             = sum_i E_z[log N(x_i; beta * z, sigma^2 * I)]
#                    (approx) = sum_i (1 / J) sum_j log N(x_i; beta * z_j, sigma^2 * I)
#
# where we use monte carlo estimates of the log lik under the prior for z to compute
# in turn, monte carlo estimates of the overall likelihood.
#
# Note that we are implicitly marginalizing z out here.
#
# We should expect this to be very slow...
#


def mle_estimate_batch_likelihood(batch, mle_params, sub_B, test_noise=None):
    '''Compute batch likelihood under naive method

    Args:
        batch: (torch.autograd.Variable) the batch of inputs X
        mle_params: (MLE_PARAMS torch.autograd.Variable tuple) the variables needed to compute
        sub_B: (int) the size of the batches used for monte carlo estimates of E_z[log lik(x)]
        test_noise: (torch.autograd.Variable) optional; introduce noise via function input instead
                    of randomly generated with in the function

    Returns:
        (torch.autograd.Variable) the marginal likelihood of the batch; shape (1, )
    '''
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
    noise = utils.make_torch_variable(np.random.randn(sub_B * B, K), False)
    if test_noise is not None:  # For debugging, allow insertion of a deterministic noise variable
        utils.check_autograd_variable_size(test_noise, [(sub_B * B, K)])
        noise = test_noise

    # Expand minibatch to match shape of noise
    batch = _mle_expand_batch(batch, sub_B)

    # Construct mu and sigma & compute density
    mu = torch.mm(noise, mle_params.beta)

    identity = utils.make_torch_variable(np.identity(M), False)
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
    '''Run an update step for naive MLE method'''
    # Create minibatch
    batch = utils.select_minibatch(x, B)

    # Estimate marginal likelihood of batch
    neg_marginal_log_lik = -1 * mle_estimate_batch_likelihood(batch, mle_params, sub_B)

    # Do a backward step
    neg_marginal_log_lik.backward()

    # Update step
    utils.gradient_descent_step_parameter_tuple(mle_params, learning_rate)

    # Constrain sigma
    mle_params.sigma.data[0] = max(1e-10, mle_params.sigma.data[0])

    # Clear gradients
    utils.clear_gradients_parameter_tuple(mle_params)

    return mle_params, neg_marginal_log_lik


def mle_forward_step_w_optim(x, mle_params, B, sub_B, optimizer):
    '''Run an update step for naive MLE method

    Args:
        x: (numpy array like) the inputs; shape (N, M)
        mle_params: (MLE_PARAMS torch.autograd.Variable tuple) the variables needed to compute
        B: (int) the batch size
        sub_B: (int) the sub-batch size (used to estimate E_z[log lik(x)])
        optimizer: the pytorch optimization routine to use

    Returns:
        the parameters, now updated & the negative marginal log likelihood (as comp'd on batch)
    '''
    # Create minibatch
    batch = utils.select_minibatch(x, B)

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


#
# Method 2:
#
# In this less-naive approach, we analytically marginalize out z instead of
# estimating the marginalized density, e.g.,
#
#    log P(x | beta, sigma^2) = sum_i log P(x_i | beta, sigma)
#                             = sum_i log N(x_i; 0, K)
#
# where K = t(B) * B + sigma^2 * I.
#

def compute_var(beta, sigma):
    '''Computes M = t(W) * W + sigma^2 * I, which is a commonly used quantity'''
    _, M = beta.size()
    identity = utils.make_torch_variable(np.identity(M), False)
    a1 = torch.mm(beta.t(), beta)
    a2 = torch.mul(sigma ** 2, identity)
    return torch.add(a1, a2)


def mle_estimate_batch_likelihood_v2(batch, mle_params):
    if not isinstance(mle_params, MLE_PARAMS):
        raise ValueError('Input params must be of type MLE_PARAMS')

    B, M = batch.size()

    mu = utils.make_torch_variable(np.zeros(M), False)
    sigma = compute_var(mle_params.beta, mle_params.sigma)

    approx_marginal_log_likelihood = mvn.torch_mvn_density(batch, mu, sigma, log=True)

    return approx_marginal_log_likelihood.sum()


def mle_forward_step_w_optim_v2(x, mle_params, B, optimizer):
    # Create minibatch
    batch = utils.select_minibatch(x, B)

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


#
# Method 3:
#
# In an alternative approach, we marginalize out beta (using a N(0, alpha^2 * I)
# prior on beta and instead try to optimize
#
#     log P(x | z, alpha, sigma) = sum_i log P(x_i | z, alpha, sigma)
#                                = sum_i log N(x_i; 0, K)
#
# where K = alpha^2 * t(Z) * Z + sigma^2 * I.
#

MLE_PARAMS_2 = namedtuple('MLE_PARAMS_2', ['z', 'sigma', 'alpha'])


def mle_initialize_parameters_v3(N, M, K):
    '''Initialize the parameters before fitting'''
    z = utils.make_torch_variable(np.random.randn(N, K), True)
    sigma = utils.make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    alpha = utils.make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
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
    identity = utils.make_torch_variable(np.identity(B), False)
    var = torch.add(
        torch.mul(mle_params_2.alpha ** 2, dot),
        torch.mul(mle_params_2.sigma ** 2, identity)
    )

    # ### Compute log lik
    mu = utils.make_torch_variable(np.zeros(B), requires_grad=False)
    approx_marginal_log_likelihood = mvn.torch_mvn_density(batch_x.t(), mu, var, log=True).sum()

    return approx_marginal_log_likelihood


def mle_forward_step_w_optim_v3(x, mle_params, B, optimizer):
    # Create minibatch
    N, _ = x.size()
    batch_ix = np.random.choice(range(N), B, replace=True)

    # Estimate marginal likelihood of batch
    neg_marginal_log_lik = -1 * mle_estimate_batch_likelihood_v3(x, batch_ix, mle_params)

    # Do a backward step
    neg_marginal_log_lik.backward()

    # Do a step
    optimizer.step()

    # Constrain sigma
    mle_params.sigma.data[0] = max(1e-10, mle_params.sigma.data[0])
    mle_params.alpha.data[0] = max(1e-10, mle_params.alpha.data[0])

    # Clear gradients
    optimizer.zero_grad()

    return mle_params, neg_marginal_log_lik


# ### Expectation maximization methods

#
# EM *should* be a lot more efficient than maximum likelihood at learning latent variable models
#
# Note that with EM, we try to optimize P(x | beta, sigma), using the iterative process to help
# with the marginalization of z. I imagine there is a way to reframe it as P(x | alpha, sigma, z)
# but have not put in the work for that yet :P
#
# The two steps of EM run as follows:
#    1) Compute first and second moments of P(z | x, beta, sigma) using last iteration's values
#    2) Use ^^^ to write the full data likelihood and maximize with respect to beta & sigma
#       (treating the moments computed in [1] as fixed)
#

EM_PARAMS = namedtuple('EM_PARAMS', ['beta', 'sigma'])


def em_initialize_parameters(M, K):
    '''Initialize the parameters before fitting'''
    beta = utils.make_torch_variable(np.random.randn(K, M), True)
    sigma = utils.make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    return EM_PARAMS(beta=beta, sigma=sigma)


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
    beta_t_ = em_params.beta.t().unsqueeze(0).expand(B, M, K)  # K x M --> M x K --> B x M x K
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

    full_data_log_likelihood = -1 * (a_1 + a_2 + a_3 + a_4 + a_5).sum()

    return full_data_log_likelihood


def em_forward_step(x, em_params, B, optimizer):
    # Create minibatch
    batch = utils.select_minibatch(x, B)

    # Compute posterior
    e_y, e_y2 = em_compute_posterior(batch, em_params)

    # Use posterior to construct full data log likelihood as function of em_params
    neg_full_data_log_likelihood = -1 * em_compute_full_data_log_likelihood(batch, em_params, e_y, e_y2)

    # Do a backward step
    neg_full_data_log_likelihood.backward()

    # Update step
    optimizer.step()

    # Constrain sigma
    em_params.sigma.data[0] = max(1e-10, em_params.sigma.data[0])

    # Clear gradients
    optimizer.zero_grad()

    return em_params, neg_full_data_log_likelihood


# ### Variational bayes methods

#
# We use a simple variational approximation here in imitation of the VAE's methodology, e.g.,
# we try to minimize the KL-divergence between the true posterior
#
#     P(z | x, beta, sigma)
#
# and the proposal distribution
#
#     q(z | x, beta_q, sigma_q) = N(x; beta_q, sigma_q^2 * I)
#
# This should be a biased estimator & ought disagree with the E-M estimate; more a proof
# of concept than anything else.
#

VB_PARAMS = namedtuple('VB_PARAMS', ['beta', 'sigma', 'beta_q', 'sigma_q'])


def vb_initialize_parameters(M, K):
    beta = utils.make_torch_variable(np.random.randn(K, M), True)
    sigma = utils.make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    beta_q = utils.make_torch_variable(np.random.randn(M, K), True)
    sigma_q = utils.make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    return VB_PARAMS(beta=beta, sigma=sigma, beta_q=beta_q, sigma_q=sigma_q)


def _reparametrize_noise(batch, noise, vb_params):
    mu = torch.mm(batch, vb_params.beta_q)

    _, K = noise.size()
    identity = utils.make_torch_variable(np.identity(K), False)
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
    identity_x = utils.make_torch_variable(np.identity(M), False)
    sigma_x = torch.mul(vb_params.sigma ** 2, identity_x)

    mu_q = torch.mm(batch, vb_params.beta_q)
    identity_q = utils.make_torch_variable(np.identity(K), False)
    sigma_q = torch.mul(vb_params.sigma_q ** 2, identity_q)

    mu_prior = utils.make_torch_variable(np.zeros(K), False)
    sigma_prior = utils.make_torch_variable(np.identity(K), False)

    # Compute log likelihoods
    log_posterior = mvn.torch_mvn_density(noise, mu_q, sigma_q, log=True)
    log_likelihood = mvn.torch_mvn_density(batch, mu_x, sigma_x, log=True)
    log_prior = mvn.torch_mvn_density(noise, mu_prior, sigma_prior, log=True)

    lower_bound = log_posterior - log_likelihood - log_prior

    return lower_bound.sum()


def vb_forward_step(x, vb_params, B, learning_rate):
    # Create minibatch
    batch = utils.select_minibatch(x, B)

    # Sample noise
    K, _ = vb_params.beta.size()
    noise = utils.make_torch_variable(np.random.randn(B, K), False)
    noise = _reparametrize_noise(batch, noise, vb_params)

    # Estimate marginal likelihood of batch
    neg_lower_bound = vb_estimate_lower_bound(batch, noise, vb_params)

    # Do a backward step
    neg_lower_bound.backward()

    # Update step
    utils.gradient_descent_step_parameter_tuple(vb_params, learning_rate)

    # Constrain sigma
    vb_params.sigma.data[0] = max(1e-10, vb_params.sigma.data[0])
    vb_params.sigma_q.data[0] = max(1e-10, vb_params.sigma_q.data[0])

    # Clear gradients
    utils.clear_gradients_parameter_tuple(vb_params)

    return vb_params, neg_lower_bound


def vb_forward_step_w_optim(x, vb_params, B, optimizer):
    # Create minibatch
    batch = utils.select_minibatch(x, B)

    # Sample noise
    K, _ = vb_params.beta.size()
    noise = utils.make_torch_variable(np.random.randn(B, K), False)
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
