# Code to write 

x. Make sure I know how torch autodifferentiation works

    x. First, fit a normal distribution using maximum likelihood
    x. Second, fit a multivariate normal distribution using maximum likelihood

x. Understand why the reparametrization trick helps

    x. Compute gradient estimates of parameter in distribution using
    
        x. the variational approximation
        x. the reparametrization

x. Implement PPCA with MLE

    x unclear if correct...
    x should be able to implement with EM (read Tipping / Bishop)
    x also GPLVM - integrating out the coefficient matrix (Lawrence)
        x e.g., most importantly, implement PPCA with beta marginalized out
    x use adam rather than hand-implemented gradient descent
    
x. Implement PPCA with EM

x. Implement PPCA with VB

    x unclear if correct...
    x use adam rather than hand-implemented gradient descent

x. Implement GPLVM with MLE

    x take a look at https://github.com/jrg365/gpytorch to see a good implementation

x. Implement GPLVM with VB

x. Implement autoencoder

x. Implement vae-ish -- but optimized using MLE
    * skipped; don't know if it makes sense

x. Implement VAE

# Flow of the project ideation (and the experiments i need to complete it)

This paper is a survey study of the following latent variable methods:
    * PCA
    * PPCA
    * GPLVM
    * ANN
    * VAE
along with a survey of their inference methods (e.g., MLE, EM, VB).

In the process we hope to answer the following questions:
    * what do we gain as we increase the expressiveness of the encoder / decoder functions?
    * what do we gain by making the methods bayesian?
    * how stable are these methods? (e.g., reinitializing with random parameters, how likely
      are we to learn the same functions?)
    * how different are the learned representations?

We will try to answer these both qualitatively and quantitatively; for the latter we will
measure reconstruction error, and, if we can figure it out, the value provided by the
learned representation for semi-supervised learning.

Data to study these with:
    * linear example
    * xor? 4d sign product?
    * nonlinear example
    * mnist

Misc studies:
x reparametrization experiment

# Standing questions

* Do i need alpha in the RBF? seems redundant?
