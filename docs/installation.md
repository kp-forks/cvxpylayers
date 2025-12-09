# Installation

## Basic Installation

Install CVXPYlayers using pip:

```bash
pip install cvxpylayers
```

## Framework-Specific Installation

CVXPYlayers supports PyTorch, JAX, and MLX. Install with your preferred framework:

::::{tab-set}

:::{tab-item} PyTorch
```bash
pip install cvxpylayers[torch]
```
:::

:::{tab-item} JAX
```bash
pip install cvxpylayers[jax]
```
:::

:::{tab-item} MLX
```bash
pip install cvxpylayers[mlx]
```
MLX is optimized for Apple Silicon (M1/M2/M3).
:::

:::{tab-item} All Frameworks
```bash
pip install cvxpylayers[all]
```
:::

::::

## Dependencies

CVXPYlayers requires:

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.11 | Runtime |
| NumPy | >= 1.22.4 | Array operations |
| CVXPY | >= 1.7.4 | Problem specification |
| diffcp | >= 1.1.0 | Differentiable cone programming |

Framework dependencies:

| Framework | Version |
|-----------|---------|
| PyTorch | >= 2.0 |
| JAX | >= 0.4.0 |
| MLX | >= 0.27.1 |

## GPU Acceleration (CuClarabel)

For GPU-accelerated solving on NVIDIA GPUs, install the CuClarabel backend:

### Prerequisites

1. **Julia**: Install from [julialang.org](https://julialang.org/)

2. **CuClarabel**: Install the Julia package
   ```julia
   using Pkg
   Pkg.add(url="https://github.com/oxfordcontrol/Clarabel.jl", rev="CuClarabel")
   ```

3. **Python packages**:
   ```bash
   pip install juliacall cupy diffqcp
   pip install "lineax @ git+https://github.com/patrick-kidger/lineax.git"
   ```

### Usage

```python
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Create layer with CuClarabel solver
layer = CvxpyLayer(
    problem,
    parameters=[A, b],
    variables=[x],
    solver=cp.CUCLARABEL
).to("cuda")
```

## Development Installation

For contributing or development:

```bash
git clone https://github.com/cvxpy/cvxpylayers.git
cd cvxpylayers
pip install -e ".[all]"
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

## Verifying Installation

Test your installation:

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Simple test problem
x = cp.Variable(2)
A = cp.Parameter((2, 2))
b = cp.Parameter(2)
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
A_t = torch.eye(2, requires_grad=True)
b_t = torch.ones(2, requires_grad=True)
(sol,) = layer(A_t, b_t)
print(f"Solution: {sol}")
print(f"Installation successful!")
```
