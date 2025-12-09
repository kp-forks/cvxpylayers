# Quickstart

This guide walks through creating your first differentiable convex optimization layer.

## Overview

CVXPYlayers lets you:
1. Define a parametrized convex optimization problem in CVXPY
2. Wrap it as a differentiable layer
3. Compute gradients through the optimization

## Step 1: Define the Problem

First, define a parametrized convex optimization problem using CVXPY:

```python
import cvxpy as cp

# Problem dimensions
n, m = 2, 3

# Decision variable
x = cp.Variable(n)

# Parameters (values provided at runtime)
A = cp.Parameter((m, n))
b = cp.Parameter(m)

# Constraints
constraints = [x >= 0]

# Objective
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))

# Problem
problem = cp.Problem(objective, constraints)
```

## Step 2: Verify DPP Compliance

Your problem must follow [Disciplined Parametrized Programming (DPP)](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming) rules:

```python
assert problem.is_dpp(), "Problem must be DPP-compliant"
```

DPP ensures the problem structure is fixed and only parameter values change. This is required for automatic differentiation.

## Step 3: Create the Layer

Wrap the problem as a differentiable layer:

::::{tab-set}

:::{tab-item} PyTorch
```python
import torch
from cvxpylayers.torch import CvxpyLayer

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
```
:::

:::{tab-item} JAX
```python
import jax
from cvxpylayers.jax import CvxpyLayer

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
```
:::

:::{tab-item} MLX
```python
import mlx.core as mx
from cvxpylayers.mlx import CvxpyLayer

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
```
:::

::::

## Step 4: Solve and Differentiate

Pass parameter values to get solutions, then backpropagate:

::::{tab-set}

:::{tab-item} PyTorch
```python
# Create parameter tensors
A_tch = torch.randn(m, n, requires_grad=True)
b_tch = torch.randn(m, requires_grad=True)

# Forward pass: solve the optimization problem
(solution,) = layer(A_tch, b_tch)

# Backward pass: compute gradients
loss = solution.sum()
loss.backward()

print(f"Solution: {solution}")
print(f"Gradient w.r.t. A: {A_tch.grad}")
print(f"Gradient w.r.t. b: {b_tch.grad}")
```
:::

:::{tab-item} JAX
```python
# Create parameter arrays
key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
A_jax = jax.random.normal(k1, shape=(m, n))
b_jax = jax.random.normal(k2, shape=(m,))

# Forward pass
(solution,) = layer(A_jax, b_jax)

# Compute gradients
def loss_fn(A, b):
    (sol,) = layer(A, b)
    return sol.sum()

gradA, gradb = jax.grad(loss_fn, argnums=[0, 1])(A_jax, b_jax)

print(f"Solution: {solution}")
print(f"Gradient w.r.t. A: {gradA}")
print(f"Gradient w.r.t. b: {gradb}")
```
:::

:::{tab-item} MLX
```python
# Create parameter arrays
A_mx = mx.random.normal((m, n))
b_mx = mx.random.normal((m,))

# Forward pass
(solution,) = layer(A_mx, b_mx)

# Compute gradients
def loss_fn(A, b):
    (sol,) = layer(A, b)
    return sol.sum()

grad_fn = mx.grad(loss_fn, argnums=[0, 1])
gradA, gradb = grad_fn(A_mx, b_mx)

print(f"Solution: {solution}")
print(f"Gradient w.r.t. A: {gradA}")
print(f"Gradient w.r.t. b: {gradb}")
```
:::

::::

## Complete Example: Least Squares

Here's a complete example solving a constrained least squares problem:

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Problem: minimize ||Ax - b||^2 subject to x >= 0
n, m = 5, 10
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)

problem = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ x - b)),
    [x >= 0]
)
assert problem.is_dpp()

# Create layer
layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

# Training loop example
A_true = torch.randn(m, n)
x_true = torch.abs(torch.randn(n))  # Ground truth (non-negative)
b_true = A_true @ x_true

# Learnable parameters
A_learn = torch.randn(m, n, requires_grad=True)
optimizer = torch.optim.Adam([A_learn], lr=0.1)

for i in range(100):
    optimizer.zero_grad()
    (x_pred,) = layer(A_learn, b_true)
    loss = torch.sum((x_pred - x_true) ** 2)
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        print(f"Iteration {i}: loss = {loss.item():.4f}")
```

## Next Steps

- {doc}`guide/basic-usage` - Constructor options and parameter handling
- {doc}`guide/batching` - Batched execution for efficiency
- {doc}`guide/geometric-programs` - Log-log convex programs
- {doc}`examples/index` - Application examples
