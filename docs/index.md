# CVXPYlayers

**Differentiable convex optimization layers for PyTorch, JAX, and MLX**

CVXPYlayers is a Python library for constructing differentiable convex optimization layers using [CVXPY](https://www.cvxpy.org/). A convex optimization layer solves a parametrized convex optimization problem in the forward pass and computes gradients via implicit differentiation in the backward pass.

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} PyTorch
:link: api/torch
:link-type: doc

`torch.nn.Module` integration with full autograd support.
:::

:::{grid-item-card} JAX
:link: api/jax
:link-type: doc

Compatible with `jax.grad`, `jax.jit`, and `jax.vmap`.
:::

:::{grid-item-card} MLX
:link: api/mlx
:link-type: doc

Apple Silicon acceleration with MLX framework.
:::
::::

## Installation

```bash
pip install cvxpylayers
```

For framework-specific installations:

```bash
pip install cvxpylayers[torch]  # PyTorch
pip install cvxpylayers[jax]    # JAX
pip install cvxpylayers[mlx]    # MLX (Apple Silicon)
```

## Quick Example

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Define a parametrized convex optimization problem
n, m = 2, 3
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)

# Create a differentiable layer
layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

# Solve and differentiate
A_tch = torch.randn(m, n, requires_grad=True)
b_tch = torch.randn(m, requires_grad=True)
(solution,) = layer(A_tch, b_tch)
solution.sum().backward()  # Gradients flow through the optimization
```

## Features

- **Disciplined Parametrized Programming (DPP)**: Problems must follow CVXPY's DPP rules for automatic differentiation
- **GPU Acceleration**: Full GPU support with CuClarabel solver (PyTorch)
- **Batched Execution**: Solve multiple problem instances in parallel
- **Geometric Programs**: Support for log-log convex programs with `gp=True`
- **Multiple Solvers**: Clarabel, SCS, ECOS, and CuClarabel backends

## Research

This library accompanies our [NeurIPS 2019 paper](https://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf) on differentiable convex optimization layers. For an introduction, see our [blog post](https://locuslab.github.io/2019-10-28-cvxpylayers/).

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
guide/index
api/index
examples/index
```

## Citing

If you use CVXPYlayers for research, please cite:

```bibtex
@inproceedings{cvxpylayers2019,
  author={Agrawal, A. and Amos, B. and Barratt, S. and Boyd, S. and Diamond, S. and Kolter, Z.},
  title={Differentiable Convex Optimization Layers},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019},
}
```
