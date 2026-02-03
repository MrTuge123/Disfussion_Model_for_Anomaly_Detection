# Normalizing Flows — Intuition and Practical Guide

This document explains the core ideas behind normalizing flows (NFs), why they work, and how to implement them at a high level. It is written for practitioners who know basic probability, calculus, and machine learning.

---

## 1) High-level idea

- Goal: Build a flexible model p_X(x) for complex data x (images, signals, etc.) that supports two operations efficiently:
  - Evaluate log-density log p_X(x)
  - Draw samples x ~ p_X(x)

- Strategy: Start from a simple base distribution z ~ p_Z(z) (e.g., standard Normal). Transform z through an invertible, differentiable mapping f to get x = f(z). If f is invertible, we can compute p_X(x) using the change-of-variable formula.

- Normalizing flow: A sequence of simple, invertible transformations (bijectors) f = f_K ∘ f_{K-1} ∘ ... ∘ f_1. Each step is designed so both the forward and inverse are easy to compute and the (log) absolute determinant of the Jacobian is tractable.


## 2) Change-of-variable formula (the math core)

Let x = f(z) where f is invertible and differentiable, with inverse z = f^{-1}(x). Then:

p_X(x) = p_Z(z) * |det(dz/dx)| = p_Z(f^{-1}(x)) * |det(J_{f^{-1}}(x))|.

Working in log-space (numerically stable):

log p_X(x) = log p_Z(z) + log |det(dz/dx)|
          = log p_Z(f^{-1}(x)) + sum_{i=1..K} log |det(J_{f_i^{-1}}(h_i))|,

where h_i is the intermediate variable at layer i. For flows, we usually compute the forward mapping z -> x and accumulate the log-determinants in the forward direction using the Jacobian of each forward transform (with appropriate sign):

log p_X(x) = log p_Z(z) - sum_{i=1..K} log |det(J_{f_i}(u_i))|  (depending on sign conventions).

Key requirement: For each f_i, computing log |det(J_{f_i})| must be efficient.


## 3) Base distribution and sampling

- Base distribution p_Z is simple (standard Normal, logistic, etc.).
- Sampling: draw z ~ p_Z, compute x = f(z) (forward pass). Because f is a composition of tractable bijectors, sampling is straightforward.
- Density evaluation: given x, compute z = f^{-1}(x) and sum the log-dets to get log p_X(x).


## 4) Designing bijectors (trade-offs)

Each bijector must be:
- Invertible and differentiable
- Fast to compute forward and inverse (or at least inverse or forward needed for your use case)
- Have a tractable Jacobian determinant (or a determinant with a cheap reduction)

Common families:

- Additive coupling (NICE): split input x into (x_a, x_b); set y_a = x_a, y_b = x_b + m(x_a). Jacobian is triangular => det = 1 (log-det = 0) so very cheap. Inverse trivial. Lacks scaling (volume changes), so often extended.

- Affine coupling (RealNVP): y_a = x_a, y_b = x_b * s(x_a) + t(x_a). Jacobian is triangular with diagonal s(x_a) => log|det| = sum log |s|. Inverse straightforward. The scale s can be parameterized (e.g., exp of network output)

- Autoregressive flows (MADE / MAF / IAF): parameterize each dimension conditioned on previous ones. Some variants allow fast density (MAF) while others allow fast sampling (IAF). Autoregressive Jacobian is triangular so log-det is sum of diagonal terms.

- Invertible 1x1 convs / invertible linear layers (Glow): used to mix channels. Determinant is det(W) where W is small (C x C) so computing log|det| is O(C^3) but C is small (channels). Efficient implementations use LU decomposition to get O(C^2).

- Squeezing / multi-scale architectures: reshape spatial dimensions into channels to apply 1x1 convs and coupling layers more effectively (common in image flows like Glow).


## 5) Training objective

Flows are trained by maximum likelihood. For a minibatch {x_i}:

1. Compute z_i = f^{-1}(x_i) (inverse pass) and accumulate log-dets.
2. Evaluate log p_Z(z_i).
3. Maximize the mean log p_X(x_i) = log p_Z(z_i) + log|det(dz/dx)|.

Equivalently, minimize negative log-likelihood (NLL). No adversarial training or variational bounds are required — flows are exact (if implemented without approximations).


## 6) Implementation sketch (PyTorch-style pseudocode)

- Each flow step is a Module returning (y, log_abs_det_J) for forward; and optionally given y returning (x, -log_abs_det_J) for inverse.

Example high-level forward evaluation for sampling:

```
# z ~ N(0,I)
z = torch.randn(batch, dim)
x = z
logdet = 0
for layer in flow_layers:
    x, ld = layer.forward(x)   # x <- layer(x)
    logdet += ld
# x is a sample
```

Example for density evaluation given x:

```
logdet = 0
y = x
for layer in reversed(flow_layers):
    y, ld = layer.inverse(y)
    logdet += ld
log_prob = base.log_prob(y) + logdet
```

Important: sign conventions vary; be consistent (some implementations accumulate -logdet in forward). Tests on small dimensions help verify correctness.


## 7) Practical tips and gotchas

- Numerical stability: scale outputs carefully, use log-scale for s, and clamp where appropriate.
- Initialization: initialize scale layers to near-identity so training is stable (e.g., small weights or zero biases for t, s=1 initially).
- Multi-scale: splitting and factoring out parts of the representation and modeling them at coarser scales helps scaling to images and reduces compute.
- Batch size: likelihood training can be sensitive to batch statistics; use reasonable batch sizes and weight decay if needed.
- Missing expressivity: coupling layers only transform part of the input each step — alternate partitions and add permutations/1x1 convs to let information flow.


## 8) When to use flows vs alternatives

- Use flows when you need exact likelihoods and efficient sampling (e.g., density estimation, some generative modeling tasks).
- If you need extremely flexible densities but can tolerate approximate likelihoods, consider VAEs or diffusion models. Flows sometimes require many layers to match the expressivity of other models for images.


## 9) References and resources

- Dinh, Krueger, Bengio — NICE (2014)
- Dinh, Sohl-Dickstein, Bengio — RealNVP (2016)
- Kingma, Dhariwal — Glow (2018)
- Germain et al. — MADE / MAF / IAF
- Papamakarios et al. — Normalizing Flows for Probabilistic Modeling and Inference (review)

Online resources:
- Open-source implementations: `n flows`, `glow` repositories, `FrEIA` framework, PyTorch/TF example notebooks
- Tutorials: blog posts explaining change-of-variables and coupling layers


---

If you'd like, I can also:
- Add a runnable minimal PyTorch example implementing a small RealNVP-style flow (data + training loop).
- Create a Jupyter notebook that demonstrates training on a 2D toy dataset and visualizes learned densities.

File created: `docs/normalizing_flows_explained.md` in the repository root.
