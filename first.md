# Introduction

There are some things neural networks are not great for:

* learning with limited amounts of data
* providing bounds / uncertainty estimates for predictions
* in the case of classification, recognizing and learning new labels

These are tasks at which bayesian methodologies, in contrast, handle well.
This post will go into several approaches fusing bayesian methods with neural
networks, specifically focusing on those published at the Bayesian Deep Learning
workshop at NIPS 2016.

I do not believe that any of these are pressing issues; save for the ultimate
goal of a general artificial intelligence, I'm not sure any of these are
relevant issues to the practical applicability of neural networks (e.g.,
if you've got limited data, are the marginal gains of using a neural
network really worth it? if you're incredibly accurate, how necessary
are explicit uncertainty estimates, versus frequentist bounds?
won't new labels be a concern if and only if there a lot of them - in
which case we can just train away the problem?)

And in fact, people have tried to fuse bayes and neural networks before:
explicitly with what were called "bayesian neural networks" - an effort
started in the 1990s and abandoned by the mid 2000s.

However, at least one case suggests that there may be something yet in this
fusion; the paper *Semi-Supervised Learning with Deep Generative Models* 
by Durk Kingma, Danilo Rezende, Shakir Mohamed, and Max Welling demonstrates
how a generative model featuring a neural network and establishes a formidable
benchmark in semi-supervised learning. We'll begin with their paper.

# Semi-Supervised Learning with Deep Generative Models

## A Prerequisite: Variational Autoencoders

The central tool used in *Semi-Supervised Learning with Deep Generative Models*
is the *variational autoencoder*: *variational* as in variational inference and
*autoencoder* as in a neural autoencoder, the former informing the methods used
to learn the model and the latter a succinct description of the structure.

We'll focus first on the latter, as it will provide a more immediate sense of
what is going on. Remember that a neural autoencoder, at it's simplest, has 
three layers:

* an input layer
* a hidden layer (usually sparse)
* an output layer, of dimension same as the input layer

At the most basic level, the *variational autoencoder* tries to fuse bayesian
methods with neural networks by placing a prior on the distribution of values
taken by the hidden layer.

What does this accomplish? (And here is me hypothesizing a little) Well, by
placing a prior on the values taken by the hidden layer, we can more 
efficiently learn a good representation of the data (we will test this in
an experiment on MNIST in a bit). Not sure there's really much else here...

From a generative model perspective, the model is actually quite simple.
Given data $\{x_i\} \in \mathbb{R}^n$, we try to learn latent representations
of each point $\{z_i\} \in \mathbb{R}^m$ where $m < n$. Each data point $x_i$
is generated conditional on $z_i$ by some distribution $p(x_i | z_i)$.

* we specify a gaussian prior on the $z_i \sim \mathcal{N}(0, I)$
* we model $p$ as a distribution incorporating neural networks; in particular
  that $p(x_i | z_i) = \mathcal{N}(x_i; \mu(z_i, \theta), \sigma(z_i, \theta))$
  where $\mu$ and $\sigma$ are feedforward neural networks taking in input $z$
  and with (different) parameters $\theta$
* additionally, since modeling $p(x | z)$ using neural networks makes the
  posterior $p(z | x)$ intractable, we approximate $p(z | x)$ with a distribution
  $q(z | x)$ that also uses a neural network, e.g., 
  $q(z_i | x_i) = \mathcal{N}(z_i; \mu(x_i, \theta), \sigma(z_i, \theta)$

Specifying the parameters of $p(x | z)$ and $q(z | x)$ as functions of neural 
networks effectively "sandwiches" the latent space $z$ between two neural 
networks, leading to a structure analagous to that of an autoencoder.

Of course, though I say that the model is "quite simple", deriving an
algorithm to learn the parameters for the variational autoencoder requires
sophisiticated methods. However, we'll get in to those later and procrastinate
on learning the math by running an experiment to get a sense of what exactly
this model is capable of.

## An experiment: what does the variational autoencoder gain us?

In order to capture what exactly the variational autoencoder is doing better,
let's compare the latent space learned by a variational autoencoder with that
learned by a standard neural autoencoder and a gaussian process model. The 
main purpose of this experiment is two-fold:

1. what do we gain by placing a prior on the sparse inner section of a
   neural autoencoder?
2. what do we gain in expressiveness of the encoding / decoding maps?

We will evalaute the performance of said algorithms on learning latent
representations on MNIST data.

*run experiments*

## Why we care about this result

## Inference: lots of tricky math

We frame learning the parameters of the variational autoencoder as 

* maximum likelihood / maximum a posteriori inference on the global parameters
* variational inference on the latent parameters

Remember that our central goal is to compute the posterior distribution of the
latent parameters

\\[p(z | x) = \frac{p(z, x)}{p(x)} = \frac{p(x | z) p(z)}{p(x)}\\]

Since $p(x)$ is intractable in this case (why is it intractable?) we instead
settle for approximating $p(z | x)$ with another distribution $q(z | x)$; in
the case of auto-encoding variational bayes, we select $q(z | x)$ to be the
family of distributions $\mathcal{N}(z_i; \mu(x_i, \phi), \sigma(x_i, \phi))$
parametrized by $\phi$, and in which $\mu$ and $\sigma$ are neural networks.

In variational inference, we try to find the distribution in the variational
family closest to the true posterior, in this case by minimizing the 
KL-divergence, e.g. we want to find $\phi$ such that we minimize

\\[KL(q_\phi(z | x) || p(z | x)) = E[\log(q_\phi(z | x))] - E[\log(p(z | x))] 
                                = E[\log(q(z | x))] - E[\log(p(x, z))] + \log p(x)
\\]

where the expectations are over $z$. Since the evidence is incomputable (and
also constant across choices of $q$), we instead minimize the ELBO

\\[ELBO(q) = -E[\log(q(z | x))] + E[\log(p(x,z))]
          = -E[\log(q(z | x))] + E[\log(p(x|z))] + E[\log(p(z))]
          = -KL(q(z | x) || p(z)) + E[\log(p(x|z))]
\\]

In the particular case of variational autoencoders, note that the ELBO depends
not only on $\phi$, but also on the parameters $\theta$ of the generative model
(specifically, the parameters of the prior on $z$ and of the conditional 
probabilities $p(x | z)$). We therefore will going forward write it instead as

\\[ELBO(\phi, \theta) = -KL(q_\phi(z|x) || p_\theta(z)) + E[\log(p\theta(x|z))]\\]

Great! Now that we have established the objective function, we just need a way
to optimize against it. We'll do so using gradient descent on minibatches of
the observations $x$ (so-called stochastic gradient descent). The details of
gradient descent depend on which gradient descent method we choose; we'll leave
that to another time and instead devote the rest of the paper to the tricky job
of elucidating exactly how the gradients are computed.

Since we chose $q\phi(z | x) = \mathcal{N}(z; \mu(x, \phi), \sigma(x, \phi))$
and $p(z) = \mathcal{N}(z; 0, I)$, we can actually analytically compute the
KL divergence term to get the expression:

\\[ELBO(\phi, \theta) = 
    (\frac{1}{2} \sum{j=1}{J} (1 + \log(\sigma(x, \phi)_j^2) - 
                                   \mu(x, \phi)_j^2 -
                                   \sigma(x, \phi)_j^2)) +
    E\phi[\log(p\theta(x|z))]
\\]

This means, in particular, that the gradient of the KL divergence can be 
computed analytically; however, we need to compute the gradient of the latter 
term through sampling (e.g., monte carlo methods).


First, we look at the gradient with respect to $\theta$: (see my calcs on paper)
We'll approximate this integral through monte carlo estimation (e.g., sampling);
note that, to compute each sample, we only need to compute the derivative of the
log density of the conditional distribution and the log prior, the former of
which means differentiating against the neural network weights.

The gradient with respect to $\phi$ is (see calcs on paper)
This gradient (for reasons I don't really understand) has too high a variance;
we reparametrize $z$ as $z = g\phi(\epsilon, x)$ with $\epsilon \sim p(\epsilon)$
which in this case we do as $g\phi(\epsilon, x) = \mu\phi(x) + \sigma\phi(x)\epsilon$
where $\epsilon \sim \mathcal{N}(0, I)$. In particular, this allows us to write:

We can then compute the overall gradient at each step by

1. sampling a minibatch of observations
2. sampling an epsilon from the noise distribution
3. computing each of the gradients
4. following your update algorithm to update parameters given the gradients

### Experiment: demonstrate variance reduction of reparametrization trick

# References
*Auto-encoding Variational Bayes* https://arxiv.org/pdf/1312.6114v10.pdf
