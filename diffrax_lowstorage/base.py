from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from diffrax import (
    RESULTS,
    AbstractSolver,
    AbstractTerm,
    LocalLinearInterpolation,
)

_SolverState: TypeAlias = None


@eqx.filter_checkpoint
def _stage(terms, t, y, tmp, args, control, *, a_i, b_i):
    # Recompute vector-field intermediates during reverse-mode differentiation.
    k = terms.vf_prod(t, y, args, control)
    tmp_next = jtu.tree_map(lambda tmpi, ki: a_i * tmpi + ki, tmp, k)
    y_next = jtu.tree_map(lambda yi, tmpi: yi + b_i * tmpi, y, tmp_next)
    return y_next, tmp_next


@eqx.filter_checkpoint
def _low_storage_step(step_arg, *, recurrence):
    # Checkpoint the whole step as well, so its stage inputs are not retained
    # between steps. Native autodiff supports JVPs and closed-over parameters.
    terms, t0, t1, y0, args = step_arg
    a = jnp.asarray(recurrence.A)
    b = jnp.asarray(recurrence.B)
    c = jnp.asarray(recurrence.C)

    dt = t1 - t0
    control = terms.contr(t0, t1)
    ts = jnp.where(c[1:] == 1.0, t1, t0 + c[1:] * dt)

    y = y0
    tmp = jtu.tree_map(jnp.zeros_like, y0)
    for i in range(len(recurrence.B)):
        if recurrence.penultimate_stage_error and i == len(recurrence.B) - 1:
            y_penultimate = y
        a_i = 0.0 if i == 0 else a[i - 1]
        t_stage = t0 if i == 0 else ts[i - 1]
        y, tmp = _stage(terms, t_stage, y, tmp, args, control, a_i=a_i, b_i=b[i])

    if recurrence.penultimate_stage_error:
        y_error = jtu.tree_map(lambda y1i, ypeni: y1i - ypeni, y, y_penultimate)
    else:
        y_error = None
    return y, y_error, {"y0": y0, "y1": y}


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

    def to_butcher(self):
        """Convert to a :class:`diffrax.ButcherTableau`.

        For methods with ``penultimate_stage_error=True``, the embedded error
        coefficients are set to ``b_sol - b_penultimate`` (the penultimate stage
        value). Otherwise ``b_error`` is zero and no error estimate is provided.
        """
        from diffrax import ButcherTableau

        s = self.num_stages
        A = self.A
        B = self.B

        # P[i, j] = coefficient of k_j in tmp_i (the recurrence accumulator).
        # P[i, j] = prod(A[j:i])  (empty product = 1 when j == i).
        # Q[i, j] = coefficient of k_j in (y_{i+1} - y0).
        # Q[i, j] = sum_{m=j}^{i} B[m] * P[m, j]
        P = np.zeros((s, s))
        Q = np.zeros((s, s))
        for i in range(s):
            P[i, i] = 1.0
            for j in range(i):
                P[i, j] = A[i - 1] * P[i - 1, j]
            q_prev = Q[i - 1] if i > 0 else np.zeros(s)
            for j in range(i + 1):
                Q[i, j] = q_prev[j] + B[i] * P[i, j]

        a_lower = tuple(Q[i, : i + 1].copy() for i in range(s - 1))
        b_sol = Q[s - 1].copy()

        if self.penultimate_stage_error:
            b_embedded = np.append(Q[s - 2, : s - 1], 0.0)
            b_error = b_sol - b_embedded
        else:
            b_error = np.zeros(s)

        return ButcherTableau(
            c=self.C[1:].copy(),
            b_sol=b_sol,
            b_error=b_error,
            a_lower=a_lower,
        )

    @classmethod
    def from_butcher(cls, tableau) -> LowStorageRecurrence:
        """Construct from a :class:`diffrax.ButcherTableau`.

        Raises ``ValueError`` if the tableau does not have the structure required
        for a 2N Williamson representation.
        """
        if tableau.implicit:
            raise ValueError("A 2N low-storage recurrence must be explicit.")
        s = tableau.num_stages
        b_sol = np.asarray(tableau.b_sol)
        a_lower = [np.asarray(a) for a in tableau.a_lower]
        C = np.concatenate([[tableau.c1], np.asarray(tableau.c)])

        if s == 1:
            return cls(A=np.array([]), B=b_sol.copy(), C=C)

        # B[i] = diagonal of a_lower; B[s-1] = b_sol[s-1].
        B = np.empty(s)
        for i in range(s - 1):
            B[i] = a_lower[i][i]
        B[s - 1] = b_sol[s - 1]

        # Q[i] contains the weights after update i. For every later row r,
        # Q[r, j] - B[j] = A[j] * Q[r, j+1]. Use the largest denominator;
        # unlike division by B[j+1], this also handles zero update weights.
        Q = np.zeros((s, s))
        for i, row in enumerate(a_lower):
            Q[i, : i + 1] = row
        Q[-1] = b_sol
        A = np.zeros(s - 1)
        for j in range(s - 1):
            column = Q[j + 1 :, j + 1]
            r = j + 1 + np.argmax(np.abs(column))
            if Q[r, j + 1] != 0:
                A[j] = (Q[r, j] - B[j]) / Q[r, j + 1]

        # Verify every stage, including columns whose weights are all zero.
        p = np.zeros(s)
        q = np.zeros(s)
        for i in range(s):
            p *= A[i - 1] if i else 0.0
            p[i] = 1.0
            q += B[i] * p
            if not np.allclose(q, Q[i]):
                raise ValueError(
                    "Butcher tableau is not representable as a 2N low-storage method."
                )

        b_error = np.asarray(tableau.b_error)
        b_penultimate = Q[-2]
        penultimate_stage_error = bool(np.any(b_error)) and np.allclose(
            b_error, b_sol - b_penultimate
        )

        return cls(A=A, B=B, C=C, penultimate_stage_error=penultimate_stage_error)


class LowStorageSolver(AbstractSolver):
    """Minimal explicit 2N low-storage Runge--Kutta solver in Williamson form."""

    recurrence: ClassVar[LowStorageRecurrence]

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def error_order(self, terms):
        if not self.recurrence.penultimate_stage_error:
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
        y1, y_error, dense_info = _low_storage_step(
            (terms, t0, t1, y0, args),
            recurrence=self.recurrence,
        )
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
            "recurrence": solver.recurrence,
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
