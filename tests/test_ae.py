import unittest

import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import torch

from code.autoencoder import Autoencoder
from code.linear_regression_lvm import make_torch_variable


class TestAutoencoder(unittest.TestCase):

    def test_autoencoder_class(self):
        N = 3
        M = 3
        K = 2

        x = np.ones((N, M)).astype(float)
        x_var = make_torch_variable(x, False)

        # ### Test output & check gradients when no hidden layers
        model = Autoencoder(M, K, [], [])

        # Ensure parameters correctly registered (there should be 2 weight + 2 bias = 4 tensors)
        parameters = list(model.parameters())

        num_parameters = len(parameters)
        self.assertEqual(4, num_parameters)

        weight_1 = np.ones((K, M)).astype(float)
        bias_1 = np.ones((K, )).astype(float)
        weight_2 = np.ones((M, K)).astype(float)
        bias_2 = np.ones((M, )).astype(float)

        parameters[0].data = torch.Tensor(weight_1).type(torch.FloatTensor)
        parameters[1].data = torch.Tensor(bias_1).type(torch.FloatTensor)
        parameters[2].data = torch.Tensor(weight_2).type(torch.FloatTensor)
        parameters[3].data = torch.Tensor(bias_2).type(torch.FloatTensor)

        # Ensure output matches
        truth = (np.dot(x, weight_1.T) + bias_1).clip(0)
        truth = np.dot(truth, weight_2.T) + bias_2

        test = model(x_var)

        assert_array_almost_equal(truth, test.data.numpy())

        # Check that gradients not none
        loss = ((x_var - test) ** 2).mean()
        loss.backward()
        for p in parameters:
            self.assertIsNotNone(p)

        # ### Test output & check gradients when hidden layers
        hidden = 4

        model = Autoencoder(M, K, [4], [4])

        # Ensure parameters correctly registered (there should be 4 weight + 4 bias = 8 tensors)
        parameters = list(model.parameters())

        num_parameters = len(parameters)
        self.assertEqual(8, num_parameters)

        weight_1 = np.ones((hidden, M)).astype(float)
        bias_1 = np.ones((hidden, )).astype(float)
        weight_2 = np.ones((K, hidden)).astype(float)
        bias_2 = np.ones((K, )).astype(float)
        weight_3 = np.ones((hidden, K)).astype(float)
        bias_3 = np.ones((hidden, )).astype(float)
        weight_4 = np.ones((M, hidden)).astype(float)
        bias_4 = np.ones((M, )).astype(float)

        parameters[0].data = torch.Tensor(weight_1).type(torch.FloatTensor)
        parameters[1].data = torch.Tensor(bias_1).type(torch.FloatTensor)
        parameters[2].data = torch.Tensor(weight_2).type(torch.FloatTensor)
        parameters[3].data = torch.Tensor(bias_2).type(torch.FloatTensor)
        parameters[4].data = torch.Tensor(weight_3).type(torch.FloatTensor)
        parameters[5].data = torch.Tensor(bias_3).type(torch.FloatTensor)
        parameters[6].data = torch.Tensor(weight_4).type(torch.FloatTensor)
        parameters[7].data = torch.Tensor(bias_4).type(torch.FloatTensor)


        # Ensure output matches
        truth = (np.dot(x, weight_1.T) + bias_1).clip(0)
        truth = (np.dot(truth, weight_2.T) + bias_2).clip(0)
        truth = (np.dot(truth, weight_3.T) + bias_3).clip(0)
        truth = np.dot(truth, weight_4.T) + bias_4

        test = model(x_var)

        assert_array_almost_equal(truth, test.data.numpy())

        # Check that gradients not none
        loss = ((x_var - test) ** 2).mean()
        loss.backward()
        for p in parameters:
            self.assertIsNotNone(p)
