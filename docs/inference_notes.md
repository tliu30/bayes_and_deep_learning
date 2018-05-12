# Gaussian Process Latent Variable Model

What kernel should I use? For now will just use the smooth prior (don't think I
have a strong reason to use other kernels)

$f(z) \sim GP(0, K(z, z^\prime))$

$g(x) \sim GP(0, K(x, x^\prime))$

$z \sim MVN(0, 1)$

Solve for $\{z\}$ (and maybe kernel parameters) maximizing the likelihood of
this model

Likelihood for a general latent variable model, e.g., $x \sim P(x | z, \theta)$
and $z \sim P(z)$: 

$$P(x | \theta) = \int_{z \in Z} P(x | z, \theta) P(z) dz = E_{p(z)}[P(x | z,
\theta)]$$

which can be approximated by drawing samples from $P(z)$ and computing
$\frac{1}{L} \sum P(x | z, \theta)$

For the gaussian process model, this becomes drawing samples $z \sim MVN(0, 1)$
and computing $\frac{1}{L} \sum P(x = f(z) | \mu_{GP}, K_{GP}(z, z^\prime))$.

We can also compute this with variational bayes, by using a Gaussian Process
model to proxy the posterior distribution $P(z|x)$. For a general latent
variable model, the KL-divergence between the approximation of the posterior
and the real posterior looks like

$KL(q(z|x), p(z|x)) = \int_z q(z|x) \log \frac{q(z|x)}{p(z|x)} = \int_z q(z|x)
\log \frac{q(z|x)p(x)}{p(z, x)} = \int_z q(z|x) \log \frac{q(z|x)}{p(z,x)} +
\int_z q(z|x) \log p(x) = E_{q(z|x)}\left[\log\frac{q(z|x)}{p(z,x)}\right] +
\log p(x)$

Since $log p(x)$ is a constant, we can focus on minimizing the first term,
e.g., 

$E_{q(z|x)}\left[\log\frac{q(z|x)}{p(z,x)}\right] = E_{q(z|x)}\left[\log q(z|x)
- \log p(x|z) - \log p(z)\right]$

For the Gaussian process model, this looks like

$E_{q(z|x)}\left[\log N(z;0, K_g(x, x^\prime)) - \log N(x; 0, K_f(z, z^\prime))
- \log N(z; 0, 1)\right]$

When actually sampling, it is (apparently) more efficient to approximate
$q(z|x)$ as a function $\phi(\epsilon, x)$ with $\epsilon \sim N(0, 1)$. In the
gaussian process case, this looks like

$q(z|x) = \phi(\epsilon, x) = K_g(z, z^\prime) \cdot x$
