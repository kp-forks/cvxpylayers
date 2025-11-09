from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

try:
    import jax
    import jax.experimental.sparse
    import jax.numpy as jnp
    import mpax
except ImportError:
    pass


class MPAX_ctx:
    Q_idxs: jnp.ndarray
    c_slice: slice
    Q_structure: tuple[jnp.ndarray, jnp.ndarray]
    Q_shape: tuple[int, int]

    A_idxs: jnp.ndarray
    b_slice: slice
    A_structure: tuple[jnp.ndarray, jnp.ndarray]
    A_shape: tuple[int, int]

    G_idxs: jnp.ndarray
    h_slice: slice
    G_structure: tuple[jnp.ndarray, jnp.ndarray]
    G_shape: tuple[int, int]

    l: jnp.ndarray
    u: jnp.ndarray

    solver: Callable

    output_slices: list[slice]

    def __init__(
        self,
        objective_structure,
        constraint_structure,
        dims,
        lower_bounds,
        upper_bounds,
        output_slices,
        options=None,
    ):
        obj_indices, obj_ptr, (n, _) = objective_structure
        self.c_slice = slice(0, n)
        obj_csr = sp.csc_array(
            (np.arange(obj_indices.size), obj_indices, obj_ptr),
            shape=(n, n),
        ).tocsr()
        self.Q_idxs = obj_csr.data
        self.Q_structure = obj_csr.indices, obj_csr.indptr
        self.Q_shape = (n, n)

        con_indices, con_ptr, (m, np1) = constraint_structure
        assert np1 == n + 1

        # Extract indices for the last column (which contains b and h values)
        # Use indices instead of slices because sparse matrices may have reduced out
        # explicit zeros, so we need to reconstruct the full dense vectors
        self.last_col_start = con_ptr[-2]
        self.last_col_end = con_ptr[-1]
        self.last_col_indices = con_indices[self.last_col_start:self.last_col_end]
        self.m = m  # Total number of constraint rows

        con_csr = sp.csc_array(
            (np.arange(con_indices.size), con_indices, con_ptr[:-1]),
            shape=(m, n),
        ).tocsr()
        split = con_csr.indptr[dims.zero]

        self.A_idxs = con_csr.data[:split]
        self.A_structure = con_csr.indices[:split], con_csr.indptr[: dims.zero + 1]
        self.A_shape = (dims.zero, n)

        self.G_idxs = con_csr.data[split:]
        self.G_structure = con_csr.indices[split:], con_csr.indptr[dims.zero :] - split
        self.G_shape = (m - dims.zero, n)

        self.l = lower_bounds if lower_bounds is not None else -jnp.inf * jnp.ones(n)
        self.u = upper_bounds if upper_bounds is not None else jnp.inf * jnp.ones(n)

        if options is None:
            options = {}
        self.warm_start = options.pop("warm_start", False)
        assert self.warm_start is False
        algorithm = options.pop("algorithm", "raPDHG")
        if algorithm == "raPDHG":
            alg = mpax.raPDHG
        elif algorithm == "r2HPDHG":
            alg = mpax.r2HPDHG
        else:
            raise ValueError("Invalid MPAX algorithm")
        solver = alg(warm_start=self.warm_start, **options)
        self.solver = jax.jit(solver.optimize)
        self.output_slices = output_slices

    def jax_to_data(
        self,
        quad_obj_values,
        lin_obj_values,
        con_values,
    ):  # TODO: Add broadcasting  (will need jnp.tile to tile structures)
        # Sign conventions for cvxpylayers vs CVXPY solver interface:
        # cvxpylayers gets data from param_prob.reduced_A with different signs than
        # CVXPY's solver interface (data[s.A], data[s.B], etc.)
        # The correct convention is: negate b and h, but NOT A or G

        # Extract values from the last column and reconstruct dense b and h vectors.
        # The last column of the constraint matrix contains the RHS values,
        # but sparse matrices may have reduced out explicit zeros, so we need to
        # build dense arrays with zeros for missing rows.
        last_col_values = con_values[self.last_col_start:self.last_col_end]

        # Split indices based on whether rows are equalities (< dims.zero) or inequalities
        split_point = jnp.searchsorted(self.last_col_indices, self.A_shape[0])

        # Build dense b and h vectors with zeros for missing rows
        b_vals = jnp.zeros(self.A_shape[0])  # Shape: (dims.zero,)
        h_vals = jnp.zeros(self.G_shape[0])  # Shape: (m - dims.zero,)

        # Fill in the values we have
        b_row_indices = self.last_col_indices[:split_point]
        h_row_indices = self.last_col_indices[split_point:] - self.A_shape[0]  # Offset to start from 0

        b_vals = b_vals.at[b_row_indices].set(-last_col_values[:split_point])  # Negate b
        h_vals = h_vals.at[h_row_indices].set(-last_col_values[split_point:])  # Negate h

        model = mpax.create_qp(
            P := jax.experimental.sparse.BCSR(
                (quad_obj_values[self.Q_idxs], *self.Q_structure),
                shape=self.Q_shape,
            ),
            q := lin_obj_values[self.c_slice],
            A := jax.experimental.sparse.BCSR(
                (con_values[self.A_idxs], *self.A_structure),
                shape=self.A_shape,
            ),
            b := b_vals,
            G := jax.experimental.sparse.BCSR(
                (con_values[self.G_idxs], *self.G_structure),
                shape=self.G_shape,
            ),
            h := h_vals,
            self.l,
            self.u,
        )
        return MPAX_data(
            model,
            self.solver,
        )

    def torch_to_data(
        self,
        quad_obj_values,
        lin_obj_values,
        con_values,
    ):  # TODO: Add broadcasting  (will need jnp.tile to tile structures)
        return self.jax_to_data(
            jnp.array(quad_obj_values),
            jnp.array(lin_obj_values),
            jnp.array(con_values),
        )

    def solution_to_outputs(self, solution):
        return (solution.primal_solution[s] for s in self.output_slices)


@dataclass
class MPAX_data:
    model: mpax.utils.QuadraticProgrammingProblem
    solver: Callable

    def jax_solve(self):
        def solver(model):
            solution = self.solver(model)
            return solution.primal_solution, solution.dual_solution

        solution, fun = jax.vjp(solver, self.model)
        return *solution, fun

    def torch_solve(self, solver_args=None):
        import torch

        primal, dual, fun = self.jax_solve()
        # Convert JAX arrays to PyTorch tensors and add batch dimension
        # (matching DIFFCP's behavior which stacks results)
        primal_torch = torch.utils.dlpack.from_dlpack(primal).unsqueeze(0)  # Shape: (1, n)
        dual_torch = torch.utils.dlpack.from_dlpack(dual).unsqueeze(0)      # Shape: (1, m)
        return (
            primal_torch,
            dual_torch,
            fun,
        )

    def jax_derivative(self, primal, dual, fun):
        return fun((primal, dual))

    def torch_derivative(self, primal, dual, adj_batch):
        import torch

        quad, lin, con = self.jax_derivative(jnp.array(primal), jnp.array(dual), adj_batch)
        return torch.tensor(quad), torch.tensor(lin), torch.tensor(con)
