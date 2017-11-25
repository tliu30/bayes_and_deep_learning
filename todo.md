Some experiments to run to make sure I know what's going on

x. Make sure I know how torch autodifferentiation works
    x. First, fit a normal distribution using maximum likelihood
    x. Second, fit a multivariate normal distribution using maximum likelihood

x. Understand why the reparametrization trick helps
    x. Compute gradient estimates of parameter in distribution using
        x. the variational approximation
        x. the reparametrization

x. Implement PPCA with MLE
    x unclear if correct...
    * should be able to implement with EM (read Tipping / Bishop)
    * also GPLVM - integrating out the coefficient matrix (Lawrence)
    * use adam rather than hand-implemented gradient descent

x. Implement PPCA with VB
    x unclear if correct...
    * use adam rather than hand-implemented gradient descent

3. Implement GPLVM with MLE

4. Implement GPLVM with VB

5. Implement autoencoder

6. Implement vae-ish -- but optimized using MLE

7. Implement VAE
