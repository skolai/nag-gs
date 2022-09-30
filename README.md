# NAG-GS

*NAG-GS: Nesterov Accelerated Gradients with Gauss-Siedel splitting*

## Overview

**NAG-GS** is a novel, robust and accelerated stochastic optimizer that relies
on two key elements: (1) an accelerated Nesterov-like Stochastic Differential
Equation (SDE) and (2) its semi-implicit Gauss-Seidel type discretization.
For theoretical background we refer user to the original paper.

## Installation

Package installation is pretty straight forward in assumtions that a user has
already installed JAX/Optax or PyTorch.

```shell
pip install git+https://github.com/skolai/nag-gs.git
```

## Usage

As soon as this package is installed one can solve a toy quadratic problem in
JAX/Optax with NAG-GS as follows.

```python
from nag_gs import nag_gs
from optax import apply_updates
import jax, jax.numpy as jnp

@jax.value_and_grad
def fn(xs):
    return xs @ xs

params = jnp.ones(3)
opt = nag_gs(alpha=0.05, mu=1.0, gamma=1.5)
opt_state = opt.init(params)
for _ in range(200):
    loss, grads = fn(params)
    grads, opt_state = opt.update(grads, opt_state, params)
    params = apply_updates(params, grads)
print(params)  # [-6.888961e-05 -6.888961e-05 -6.888961e-05
```

The same optimization problem could be solved with NAG4 (a variant of NAG-GS
with fixed and constant scaling factor γ).

```python
from nag_gs import NAG4
import torch as T

def fn(xs):
    return xs @ xs

params = T.ones(3, requires_grad=True)
opt = NAG4([params], lr=0.05, mu=1.0, gamma=1.5)
for _ in range(200):
    loss = fn(params)
    loss.backward()
    opt.step()
    opt.zero_grad()
print(params.detach().numpy())  # [0.00029082 0.00029082 0.00029082]
```

## Citation

```bibtex
@misc{leplat2022nag,
  doi = {10.48550/arxiv.2209.14937},
  url = {https://arxiv.org/abs/2209.14937},
  author = {Leplat, Valentin and Merkulov, Daniil and Katrutsa, Aleksandr and Bershatsky, Daniel and Oseledets, Ivan},
  title = {NAG-GS: Semi-Implicit, Accelerated and Robust Stochastic Optimizers},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
