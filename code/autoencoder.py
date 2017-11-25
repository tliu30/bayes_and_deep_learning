import torch.nn as nn
import torch.nn.functional as F

from code.linear_regression_lvm import select_minibatch


class Autoencoder(nn.Module):

    def __init__(self, M, K, encoder_hidden, decoder_hidden):
        super(Autoencoder, self).__init__()
        layer_shapes = [M] + encoder_hidden + [K] + decoder_hidden + [M]
        self.layers = nn.ModuleList([
            nn.Linear(in_shape, out_shape) for in_shape, out_shape
            in zip(layer_shapes[:-1], layer_shapes[1:])
        ])

    def forward(self, x):
        input = x
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            input = F.relu(layer(input))
        out = self.layers[-1](input)
        return out


def ae_forward_step_w_optim(x, model, B, optimizer):
    # Clear gradients
    optimizer.zero_grad()

    # Create minibatch
    batch = select_minibatch(x, B)

    # Evaluate loss
    reconstruction = model(batch)
    loss = ((batch - reconstruction) ** 2).mean()

    # Backward step
    loss.backward()

    # Update step
    optimizer.step()

    return model, loss
