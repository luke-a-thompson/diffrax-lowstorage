from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, TypeAlias

import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import RESULTS, AbstractSolver, AbstractTerm, LocalLinearInterpolation
from jaxtyping import ArrayLike

_SolverState: TypeAlias = None


class TwoNSolver(AbstractSolver):
    """Minimal explicit 2N low-storage Runge--Kutta solver.

    This implements the classic Williamson-style recursion using three coefficient
    arrays:

    - `A`: length `s - 1`
    - `B`: length `s`
    - `C`: length `s` (stage-time fractions, with `C[0] == 0`)

    for `s` stages.

    It intentionally omits FSAL, embedded error estimates, and high-order dense output.
    """

    A: ArrayLike
    B: ArrayLike
    C: ArrayLike

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def init(
        self,
        terms: AbstractTerm,
        t0,
        t1,
        y0,
        args,
    ) -> _SolverState:
        del terms, t0, t1, y0, args
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0,
        t1,
        y0,
        args,
        solver_state: _SolverState,
        made_jump,
    ):
        del solver_state, made_jump
        a = jnp.asarray(self.A)
        b = jnp.asarray(self.B)
        c = jnp.asarray(self.C)
        if a.ndim != 1 or b.ndim != 1 or c.ndim != 1:
            raise ValueError("A, B, C must all be 1D arrays")
        num_stages = b.shape[0]
        if num_stages < 1:
            raise ValueError("B must contain at least one stage coefficient")
        if c.shape[0] != num_stages:
            raise ValueError("C must have the same length as B")
        if a.shape[0] != num_stages - 1:
            raise ValueError("A must have length len(B) - 1")

        dt = t1 - t0
        control = terms.contr(t0, t1)

        k0 = terms.vf_prod(t0, y0, args, control)
        tmp0 = k0
        y1 = jtu.tree_map(lambda y, tmp: y + b[0] * tmp, y0, tmp0)

        def body_fun(i, carry):
            y, tmp = carry
            ti = jnp.where(c[i] == 1, t1, t0 + c[i] * dt)
            k = terms.vf_prod(ti, y, args, control)
            tmp = jtu.tree_map(lambda tmp_i, k_i: a[i - 1] * tmp_i + k_i, tmp, k)
            y = jtu.tree_map(lambda y_i, tmp_i: y_i + b[i] * tmp_i, y, tmp)
            return y, tmp

        y1, _ = lax.fori_loop(1, b.shape[0], body_fun, (y1, tmp0))
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0,
        y0,
        args,
    ):
        return terms.vf(t0, y0, args)
