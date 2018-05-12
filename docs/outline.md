# Introduction

Learning lower-dimensional representations of observed data was one of the
first ideas in the theory of machine learning & statistics to capture
my imagination. If statistics is the way of cutting the signal away from
the noise, then lower-dimensional representations proferred a way
to cut to the truth and, in an Occam's Razor sense, simplify the world
around us.

This series is an exploration of a number of latent variable methods
varying in complexity, as I try to get an understanding of what
advantages are leant by adding complexity to these models; specifically,
we will be taking a look at:
* PCA (specifically, probabilistic PCA)
* GPLVM
* Autoencoding neural networks
* Variational autoencoders

In the process, we will generate plots to survey the representations
learned by these methods across a variety of data sets, and try
to quantify their value by comparing the value of the learned 
representations from these datasets in prediction (inspired by
the value provided by the variational autoencoder in semi-supervised
settings).

There will be no theoretical results here, but there will be code :)

This post has been organized to have subcontent in each section;
the main body of the work will continue below (highlighting the
results of each method, as well as maybe some history), while
breakout sections will discuss the implementation details of
each method (e.g., setup, inference, etc)

Additionally, value of adding target a la semi-supervised learning approach

# Datasets examined

We want to choose datasets that highlight the values of adding
complexity...so chose a couple simulated ones, and some natural ones:

* oil fields (from GPLVM paper)
* MNIST (from...everyone)
* Iris dataset (from...everyone)
* simple linear example
* xor?

For this, we compare bayes and non-bayes methods, to answer question
of how the different inference methods fare.

Scoring will be applied to the XOR case, iris dataset, and MNIST
* M1 + SVM (e.g., generative model to make z, then SVM)
* M2 (e.g., generative classification model)
* M1 stacked as input to M2
For this, we try to use variational bayes for everything - tries
to answer question of value of added complexity
