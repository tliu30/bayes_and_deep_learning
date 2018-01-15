import numpy as np
import torch
from torch.autograd import Variable

from code.utils import check_autograd_variable_size


# ### Matrix computation utilities (needed in particular to compute determinant)

class Cholesky(torch.autograd.Function):
    '''Implement forward and backward directions for Cholesky decomposition (from pytorch forums)'''
    # NOTE: Not tested right now...mostly b/c I don't know how to test the derivative

    @staticmethod
    def forward(ctx, a):
        '''If A is a Hermitian positive-definite matrix, finds L such that A = LL^*'''
        # For this, we can just use the bindings for BLAS
        l = torch.potrf(a, False)

        # Save for backward step
        ctx.save_for_backward(l)

        return l

    @staticmethod
    def backward(ctx, grad_output):
        '''Compute corresponding derivative'''
        # Retrieve L as computed in the forward step & compute it's inverse
        l, = ctx.saved_variables
        linv = l.inverse()

        # Do a lot of math
        a = torch.tril(torch.mm(l.t(), grad_output))
        b = torch.tril(1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        inner = a * b

        s = torch.mm(linv.t(), torch.mm(inner, linv))

        # could re-symmetrise 
        # s = (s+s.t())/2.0
        
        return s


def torch_determinant(square_matrix):
    '''Compute the determinant of the matrix in a autodiff-friendly way'''
    return Cholesky.apply(square_matrix).diag().prod() ** 2


def torch_log_determinant(square_matrix):
    '''Compute the determinant of the matrix in a autodiff-friendly way'''
    return 2 * torch.log(Cholesky.apply(square_matrix).diag()).sum()


# ### Multivariate normal density computation

def torch_mvn_density(x, mu, sigma, log=False):
    '''Compute multivariate normal pdf at x

    Args:
        x: (torch.Variable) dimension (B, M)
        mu: (torch.Variable) dimension (M, ), (1, M), or (B, M)
        sigma: (torch.Variable) dimension M x M
        log: (bool) if True, return log densities (default False)

    Returns:
        densities: (torch.Variable) dimension (B, )
    '''
    # Input validation
    B, M = x.data.size()
    check_autograd_variable_size(mu, [(M, ), (1, M), (B, M)])
    check_autograd_variable_size(sigma, [(M, M)])

    # Precompute functions of sigma
    mod_log_det_sigma = torch_log_determinant(2 * np.pi * sigma)
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
        normalization_constant = -0.5 * mod_log_det_sigma
        exponent = -0.5 * quad_form
        density = normalization_constant + exponent
    else:
        normalization_constant = 1.0 / torch.sqrt(torch.exp(mod_log_det_sigma))
        exponent = torch.exp(-0.5 * quad_form)
        density = normalization_constant * exponent

    return density


def _make_batch_matrices_w_diagonal(sigma, dim):
    diagonal_matrices = [torch.diag(s.expand(dim, )).unsqueeze(0) for s in sigma]
    return torch.cat(diagonal_matrices, dim=0)


def torch_diagonal_mvn_density_batch(x, mu, var, log=False):
    '''Compute multivariate normal pdf at x

    Args:
        x: (torch.Variable) dimension (B, M)
        mu: (torch.Variable) dimension (B, M)
        var: (torch.Variable) dimension (B, )
        log: (bool) if True, return log densities (default False)

    Returns:
        densities: (torch.Variable) dimension (B, )
    '''
    # Input validation
    B, M = x.data.size()
    check_autograd_variable_size(mu, [(B, M)])
    check_autograd_variable_size(var, [(B,), (B, M)])

    if var.size() != (B,):
        raise RuntimeError

    # Precompute functions of sigma
    mod_log_det_sigma = M * torch.log(2 * np.pi * var)  # (B, )
    inv_var = (var ** -1)  # (B, 1)

    # ### Compute quadratic form for exponentiated segment

    # Quadratic form
    x_min_mu = (x - mu).unsqueeze(1)  # (B, M) --> (B, 1, M)
    x_min_mu_t = (x - mu).unsqueeze(2)  # (B, M) --> (B, M, 1)
    inv_var = _make_batch_matrices_w_diagonal(inv_var, M)  # (B, ) --> (B, M, M)

    quad_form = torch.bmm(x_min_mu, inv_var)  # (B, 1, M) x (B, M, M) --> (B, 1, M)
    quad_form = torch.bmm(quad_form, x_min_mu_t)  # (B, 1, M) x (B, M, 1) --> (B, 1, 1)
    quad_form = quad_form.squeeze(2).squeeze(1)  # (B, 1, 1) --> (B, )

    # Compute total density, computing log density if requested
    if log:
        normalization_constant = -0.5 * mod_log_det_sigma
        exponent = -0.5 * quad_form
        density = normalization_constant + exponent
    else:
        normalization_constant = 1.0 / torch.sqrt(torch.exp(mod_log_det_sigma))
        exponent = torch.exp(-0.5 * quad_form)
        density = normalization_constant * exponent

    return density
