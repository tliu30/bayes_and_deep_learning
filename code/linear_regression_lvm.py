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

logger = logging.getLogger(__name__)


def make_torch_variable(value, requires_grad, dtype=torch.FloatTensor):
    if not isinstance(value, torch.Tensor):
        value = torch.Tensor(value)
    return Variable(value.type(dtype), requires_grad=requires_grad)


def initialize_parameters(M, K):
    '''Initialize the parameters before fitting'''
    beta_guess = make_torch_variable(np.random.randn(M, K), True)
    sigma_guess = make_torch_variable(np.random.rand(1) * 10 + 1e-10, True)
    return beta_guess, sigma_guess


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
        logger.debug('Updating {name:s} (value, grad, step) = ({v:.2f}, {g:.2f}, {s:.2f}'
                     .format(name=nm, v=variable.data, g=variable.grad, s=learning_rate))
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


def _mle_expand_batch(variable, sub_B):
    '''Reshape (B, K) to (sub_B * B, K) such that (b + sub_b * B, k) = (b, k)'''
    B, K = variable.size()
    return variable.expand(sub_B, B, K).contiguous().view(sub_B * B, K)


def _mle_unpack_likelihood(variable, sub_B, B):
    '''Unpack likelihood from (N * M, ) to (N, M) s.t. (n + m * N, ) --> (n, m)'''
    mvn.check_autograd_variable_size(variable, [(sub_B * B, )])
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

    mvn.check_autograd_variable_size(mle_params.sigma, [(1, )])

    # ### Computation

    # Sample noise required to compute monte carlo estimate of likelihood
    noise = make_torch_variable(np.random.randn(sub_B * B, K), False)
    if test_noise is not None:  # For debugging, allow insertion of a deterministic noise variable
        mvn.check_autograd_variable_size(test_noise, [(sub_B * B, K)])
        noise = test_noise

    # Expand minibatch to match shape of noise
    batch = _mle_expand_batch(batch, sub_B)

    # Construct mu and sigma & compute density
    mu = torch.mm(noise, mle_params.beta)

    identity = make_torch_variable(np.identity(M), False)
    sigma = torch.mul(mle_params.sigma, identity)

    likelihood = mvn.torch_mvn_density(batch, mu, sigma)

    # Reshape density to (sub_B, B) and sum across first dimension
    mvn.check_autograd_variable_size(likelihood, [(sub_B * B, )])
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

    # Clear gradients
    clear_gradients_parameter_tuple(mle_params)

    return mle_params, learning_rate
