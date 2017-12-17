import logging

import numpy as np
import torch
from torch.autograd import Variable

logger = logging.getLogger(__name__)


# ### Autograd variable helpers

def check_autograd_variable_size(variable, size_tuples):
    '''Assert that variable's size is one of several possible accepted shapes

    Args:
        variable: (torch.autograd.Variable)
        size_tuples: (iter of tuples) possible permitted sizes, e.g., [(3, 2), (2, 3)]
    '''
    size_arrays = [np.array(tup) for tup in size_tuples]
    found_size = variable.data.size()
    matches = [all(found_size == array) for array in size_arrays]
    if not any(matches):
        raise ValueError('expected {} but found {}'.format(size_tuples, found_size))


def make_torch_variable(value, requires_grad, dtype=torch.FloatTensor):
    '''Create a torch autograd variable from a numpy array or torch Tensor

    Used to provide a short hand for variable creation

    Args:
        value: (np.array or torch.Tensor) converts numpy array to torch Tensor if needed
        requires_grad: (bool) whether or not to track the gradient

    Returns:
        (torch.autograd.Tensor)
    '''
    if not isinstance(value, torch.Tensor):
        value = torch.Tensor(value)
    return Variable(value.type(dtype), requires_grad=requires_grad)


def select_minibatch(x, B, replace=True):
    '''Shorthand for selecting B random elements & converting to autograd Variable

    Args:
        x: (np.array) 2-d array
        B: (int) the batch size
        replace: (bool) whether or not to sample with replacement

    Returns:
        (torch.autograd.Variable) with dimension (B, _)
    '''
    # Remember, we assume that we're using row vectors!
    N, _ = x.shape
    ix_mini = np.random.choice(range(N), B, replace=replace)
    return make_torch_variable(x[ix_mini, :], False)


# ### Gradient helpers

def gradient_descent_step(variable, learning_rate):
    '''Perform gradient update on variable'''
    # TODO: deprecate manual implementation of gradient descent in favor of pytorch optim
    if variable.grad is None:
        raise ValueError('no gradients')

    if variable.grad.data.numpy().sum() == 0:
        raise ValueError('gradients look zeroed out...')

    variable.data = variable.data - (learning_rate * variable.grad.data)

    return variable


def gradient_descent_step_parameter_tuple(parameter_tuple, learning_rate):
    '''Gradient descent across a tuple of parameters'''
    # TODO: deprecate manual implementation of gradient descent in favor of pytorch optim
    for nm in parameter_tuple._fields:
        variable = parameter_tuple.__getattribute__(nm)
        logger.debug('Updating {name:s} (value, grad, step) = ({v:s}, {g:s}, {s:s}'
                     .format(name=nm, v=str(variable.data), g=str(variable.grad),
                             s=str(learning_rate)))
        gradient_descent_step(variable, learning_rate)
    return parameter_tuple


def clear_gradients(variable):
    '''Zero all gradients'''
    # TODO: deprecate manual implementation of gradient descent in favor of pytorch optim
    if variable.grad is None:
        raise ValueError('no gradients')

    variable.grad.data.zero_()
    return variable


def clear_gradients_parameter_tuple(parameter_tuple):
    '''Zero all gradients across a tuple of paramterse'''
    for nm in parameter_tuple._fields:
        variable = parameter_tuple.__getattribute__(nm)
        clear_gradients(variable)
