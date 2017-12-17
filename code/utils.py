import numpy as np


def check_autograd_variable_size(variable, size_tuples):
    '''Assert that variable's size is one of several possible accepted shapes'''
    size_arrays = [np.array(tup) for tup in size_tuples]
    found_size = variable.data.size()
    matches = [all(found_size == array) for array in size_arrays]
    if not any(matches):
        raise ValueError('expected {} but found {}'.format(size_tuples, found_size))
