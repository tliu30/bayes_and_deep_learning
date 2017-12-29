import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from code.utils import select_minibatch


class VAE(nn.Module):
    def __init__(self, m1, m2, n_hidden):
        super(VAE, self).__init__()

        # Encoders
        self.fc1 = nn.Linear(m1, n_hidden)
        self.fc21 = nn.Linear(n_hidden, m2)
        self.fc22 = nn.Linear(n_hidden, m2)

        # Decoders
        self.fc3 = nn.Linear(m2, n_hidden)
        self.fc4 = nn.Linear(n_hidden, m1)

        # Dims
        self.m1 = m1
        self.m2 = m2
        self.n_hidden = n_hidden

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3)  # if doing classification, this ought to be a sigmoid-like

    def forward(self, x):
        # Use encoder to compute mean and log variance of latent variables
        mu, logvar = self.encode(x.view(-1, self.m1))

        # Reparametrize in to latent variable z through a sampling step
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, m1, batch_size):
    BCE = F.mse_loss(recon_x, x.view(-1, m1))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * m1

    return BCE + KLD, KLD


def train(x, model, optimizer, batch_size):
    model.train()
    train_loss = 0

    batch = select_minibatch(x, batch_size, replace=False)
    optimizer.zero_grad()

    recon_batch, mu, logvar = model(batch)
    loss, kld = loss_function(recon_batch, batch, mu, logvar, model.m1, batch_size)
    loss.backward()

    train_loss += loss.data[0]

    optimizer.step()

    return model, loss, kld


def partial_train(x, model, batch_size):
    model.train()

    batch = select_minibatch(x, batch_size, replace=False)
    recon_batch, mu, logvar = model(batch)

    return recon_batch, mu, logvar
