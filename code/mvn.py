import numpy as np
import torch
from torch.autograd import Variable


class Cholesky(torch.autograd.Function):
    '''Stolen from py torch forums'''

    @staticmethod
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l

    @staticmethod
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        
        linv =  l.inverse()
        
        inner = torch.tril(torch.mm(l.t(),grad_output))*torch.tril(1.0-Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))

        # could re-symmetrise 
        # s = (s+s.t())/2.0
        
        return s


def torch_determinant(square_matrix):
    return Cholesky.apply(square_matrix).diag().prod() ** 2


def check_autograd_variable_size(variable, size_tuples):
    size_arrays = [np.array(tup) for tup in size_tuples]
    found_size = variable.data.size()
    matches = [all(found_size == array) for array in size_arrays]
    if not any(matches):
        raise ValueError('expected {} but found {}'.format(size_tuples, found_size))


def torch_mvn_density(x, mu, sigma, log=False):
    '''Compute multivariate normal pdf at x
    
    Args:
        x: (torch.Variable) dimension (B, M)
        mu: (torch.Variable) dimension (M, ), (1, M), or (B, M)
        sigma: (torch.Variable) dimension M x M 

    Returns:
        densities: (torch.Variable) dimension
    '''
    # Input validation
    B, M = x.data.size()
    check_autograd_variable_size(mu, [(M, ), (1, M), (B, M)])
    check_autograd_variable_size(sigma, [(M, M)])

    # Precompute functions of sigma
    mod_det_sigma = torch_determinant(2 * np.pi * sigma)
    inv_sigma = torch.inverse(sigma)

    # ### Compute quadratic form for exponentiated segment

    # Quadratic form
    mu = mu.expand_as(x)  # (1, M) --> (B, M)
    x_min_mu = (x - mu).unsqueeze(1)  # (B, M) --> (B, 1, M)
    x_min_mu_t = (x - mu).unsqueeze(2)  # (B, M) --> (B, M, 1)
    inv_sigma = inv_sigma.expand(B, M, M)  # (M, M) --> (B, M, M)

    quad_form = torch.bmm(x_min_mu, inv_sigma)  # (B, 1, M) x (B, M, M) --> (B, 1, M)
    quad_form = torch.bmm(quad_form, x_min_mu_t)  # (B, 1, M) x (B, M, 1) --> (B, 1, 1)
    quad_form = quad_form.squeeze(2).squeeze(1)  # (B, 1, 1) --> (B, )

    # Compute total density, computing log density if requested
    if log:
        normalization_constant = -0.5 * torch.log(mod_det_sigma)
        exponent = -0.5 * quad_form
        density = normalization_constant + exponent
    else:
        normalization_constant = 1.0 / torch.sqrt(mod_det_sigma)
        exponent = torch.exp(-0.5 * quad_form)
        density = normalization_constant * exponent

    return density


def torch_gp_mvn_log_density(x, sigma, sigma_det=None, sigma_inv=None):
    '''In GPs, we want to sample across the columns (e.g., features) rather than rows

    This should be equivalent to computing the torch_mvn_density, but i'm failing atm
    '''
    _, m = x.size()

    sigma_det = torch_determinant(sigma) if sigma_det is None else sigma_det
    sigma_inv = torch.inverse(sigma) if sigma_inv is None else sigma_inv

    a_1 = m * torch.log(sigma_det)
    a_2 = torch.mm(sigma_inv, torch.mm(x, x.t())).diag().sum()

    log_likelihood = -0.5 * (a_1 + a_2)

    return log_likelihood
