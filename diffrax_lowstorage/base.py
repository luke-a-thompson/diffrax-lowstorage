from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from diffrax import RESULTS, AbstractSolver, AbstractTerm, LocalLinearInterpolation

_SolverState: TypeAlias = None


@dataclass(frozen=True)
class LowStorageRecurrence:
    """Coefficients for a 2N Williamson-style low-storage Runge--Kutta method."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    penultimate_stage_error: bool = False

    num_stages: int = field(init=False)

    def __post_init__(self):
        a = np.asarray(self.A)
        b = np.asarray(self.B)
        c = np.asarray(self.C)
        if a.ndim != 1 or b.ndim != 1 or c.ndim != 1:
            raise ValueError("A, B, C must all be 1D arrays")
        num_stages = b.shape[0]
        if num_stages < 1:
            raise ValueError("B must contain at least one stage coefficient")
        if c.shape[0] != num_stages:
            raise ValueError("C must have the same length as B")
        if a.shape[0] != num_stages - 1:
            raise ValueError("A must have length len(B) - 1")
        if self.penultimate_stage_error and num_stages < 2:
            raise ValueError(
                "Need at least two stages for `use_penultimate_stage_error=True`."
            )
        object.__setattr__(self, "num_stages", num_stages)


class LowStorageSolver(AbstractSolver):
    """Minimal explicit 2N low-storage Runge--Kutta solver in Williamson form."""

    recurrance: ClassVar[LowStorageRecurrence]

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def error_order(self, terms):
        if not self.recurrance.penultimate_stage_error:
            return None

        # For these 2N methods, penultimate stage is taken to be order-1, so the local
        # order of the embedded error estimate is equal to `order`.
        return self.order(terms)

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
        a = jnp.asarray(self.recurrance.A)
        b = jnp.asarray(self.recurrance.B)
        c = jnp.asarray(self.recurrance.C)

        dt = t1 - t0
        control = terms.contr(t0, t1)

        # Precompute stage times outside the scan
        ts = jnp.where(c[1:] == 1.0, t1, t0 + c[1:] * dt)

        tmp = terms.vf_prod(t0, y0, args, control)
        y1 = jtu.tree_map(lambda y, t: y + b[0] * t, y0, tmp)

        def body_fun(carry, coeffs):
            y, tmp = carry
            a_i, b_i, ti = coeffs
            tmp = jtu.tree_map(
                lambda t, k: a_i * t + k, tmp, terms.vf_prod(ti, y, args, control)
            )
            y = jtu.tree_map(lambda yi, t: yi + b_i * t, y, tmp)
            return (y, tmp), None

        if self.recurrance.penultimate_stage_error:
            # Run scan up to the penultimate stage, then do the final stage manually.
            # This keeps the carry at true 2N (y, tmp) — no extra state copy needed.
            (y_pen, tmp_pen), _ = lax.scan(
                body_fun, (y1, tmp), (a[:-1], b[1:-1], ts[:-1])
            )
            tmp_pen = jtu.tree_map(
                lambda t, k: a[-1] * t + k,
                tmp_pen,
                terms.vf_prod(ts[-1], y_pen, args, control),
            )
            y1 = jtu.tree_map(lambda yi, t: yi + b[-1] * t, y_pen, tmp_pen)
            y_error = jtu.tree_map(lambda yf, yp: yf - yp, y1, y_pen)
        else:
            (y1, _), _ = lax.scan(body_fun, (y1, tmp), (a, b[1:], ts))
            y_error = None

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0,
        y0,
        args,
    ):
        return terms.vf(t0, y0, args)

    def to_commutator_free(self):
        try:
            from georax import AbstractLowStorageCommutatorFreeSolver
        except ImportError as exc:
            raise ImportError(
                "georax is required to convert a low-storage solver into a "
                "commutator-free solver."
            ) from exc

        solver = self

        def order(self, terms):
            return solver.order(terms)

        cls_dict = {
            "recurrence": solver.recurrance,
            "order": order,
            "__module__": type(self).__module__,
        }

        if hasattr(type(self), "antisymmetric_order"):
            def antisymmetric_order(self, terms):
                return solver.antisymmetric_order(terms)

            cls_dict["antisymmetric_order"] = antisymmetric_order

        commutator_free_cls = type(
            f"{type(self).__name__}CommutatorFree",
            (AbstractLowStorageCommutatorFreeSolver,),
            cls_dict,
        )
        return commutator_free_cls()
