from __future__ import annotations

import functools as ft
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


def _materialise_tree(primal, grad_primal):
    if grad_primal is None:
        return jtu.tree_map(jnp.zeros_like, primal)
    return jtu.tree_map(
        lambda p, g: jnp.zeros_like(p) if g is None else g, primal, grad_primal
    )


def _any_perturbed(tree):
    return any(jtu.tree_leaves(tree))


def _none_tree(tree):
    return jtu.tree_map(lambda _: None, tree)


def _first_stage(terms, t, y, args, control, *, b0):
    tmp = terms.vf_prod(t, y, args, control)
    y_next = jtu.tree_map(lambda yi, tmpi: yi + b0 * tmpi, y, tmp)
    return y_next, tmp


def _later_stage(terms, t, y, tmp, args, control, *, a_i, b_i):
    k = terms.vf_prod(t, y, args, control)
    tmp_next = jtu.tree_map(lambda tmpi, ki: a_i * tmpi + ki, tmp, k)
    y_next = jtu.tree_map(lambda yi, tmpi: yi + b_i * tmpi, y, tmp_next)
    return y_next, tmp_next


def _run_low_storage_step(terms, t0, t1, y0, args, *, recurrence):
    a = jnp.asarray(recurrence.A)
    b = jnp.asarray(recurrence.B)
    c = jnp.asarray(recurrence.C)

    dt = t1 - t0
    control = terms.contr(t0, t1)
    ts = jnp.where(c[1:] == 1.0, t1, t0 + c[1:] * dt)

    stage_inputs_y = [y0]
    stage_inputs_tmp = [None]

    y, tmp = _first_stage(terms, t0, y0, args, control, b0=b[0])
    for i, (a_i, b_i, t_stage) in enumerate(zip(a, b[1:], ts), start=1):
        stage_inputs_y.append(y)
        stage_inputs_tmp.append(tmp)
        y, tmp = _later_stage(terms, t_stage, y, tmp, args, control, a_i=a_i, b_i=b_i)

    if recurrence.penultimate_stage_error:
        y_error = jtu.tree_map(lambda y1i, ypeni: y1i - ypeni, y, stage_inputs_y[-1])
    else:
        y_error = None
    dense_info = dict(y0=y0, y1=y)
    return (y, y_error, dense_info), (
        control,
        ts,
        stage_inputs_y,
        stage_inputs_tmp,
        tmp,
    )


@eqx.filter_custom_vjp
def _low_storage_step(vjp_arg, *, recurrence):
    terms, t0, t1, y0, args = vjp_arg
    out, _ = _run_low_storage_step(terms, t0, t1, y0, args, recurrence=recurrence)
    return out


@_low_storage_step.def_fwd
def _low_storage_step_fwd(perturbed, vjp_arg, *, recurrence):
    del perturbed
    terms, t0, t1, y0, args = vjp_arg
    out, _ = _run_low_storage_step(terms, t0, t1, y0, args, recurrence=recurrence)
    return out, None


@_low_storage_step.def_bwd
def _low_storage_step_bwd(residuals, grad_out, perturbed, vjp_arg, *, recurrence):
    del residuals
    terms, t0, t1, y0, args = vjp_arg
    terms_perturbed, t0_perturbed, t1_perturbed, _, args_perturbed = perturbed
    (
        (y1, y_error, dense_info),
        (
            control,
            ts,
            stage_inputs_y,
            stage_inputs_tmp,
            tmp_final,
        ),
    ) = _run_low_storage_step(terms, t0, t1, y0, args, recurrence=recurrence)
    grad_y1, grad_y_error, grad_dense_info = grad_out

    diff_terms = eqx.filter(terms, eqx.is_inexact_array)
    diff_args = eqx.filter(args, eqx.is_inexact_array)
    grad_terms = jtu.tree_map(jnp.zeros_like, diff_terms)
    grad_args = jtu.tree_map(jnp.zeros_like, diff_args)
    grad_control = jtu.tree_map(
        jnp.zeros_like, eqx.filter(control, eqx.is_inexact_array)
    )
    grad_t0 = jnp.zeros_like(t0)
    grad_t1 = jnp.zeros_like(t1)

    gdi = grad_dense_info or {}
    grad_dense_y0 = _materialise_tree(dense_info["y0"], gdi.get("y0"))
    grad_dense_y1 = _materialise_tree(dense_info["y1"], gdi.get("y1"))

    grad_y = _materialise_tree(y1, grad_y1)
    grad_y = eqx.apply_updates(grad_y, grad_dense_y1)
    if recurrence.penultimate_stage_error:
        grad_y_error = _materialise_tree(y_error, grad_y_error)
        grad_y = eqx.apply_updates(grad_y, grad_y_error)

    grad_tmp = jtu.tree_map(jnp.zeros_like, tmp_final)

    for i in range(len(recurrence.B) - 1, 0, -1):
        y_in = stage_inputs_y[i]
        tmp_in = stage_inputs_tmp[i]
        assert tmp_in is not None
        t_stage = ts[i - 1]
        a_i = recurrence.A[i - 1]
        b_i = recurrence.B[i]
        _, pullback = eqx.filter_vjp(
            ft.partial(_later_stage, a_i=a_i, b_i=b_i),
            terms,
            t_stage,
            y_in,
            tmp_in,
            args,
            control,
        )
        dterms, d_t_stage, grad_y, grad_tmp, dargs, dcontrol = pullback(
            (grad_y, grad_tmp)
        )
        if recurrence.penultimate_stage_error and i == len(recurrence.B) - 1:
            grad_y = jtu.tree_map(lambda gy, ge: gy - ge, grad_y, grad_y_error)
        if t0_perturbed or t1_perturbed:
            c_i = recurrence.C[i]
            if t0_perturbed:
                grad_t0 = grad_t0 + (1.0 - c_i) * d_t_stage
            if t1_perturbed:
                grad_t1 = grad_t1 + c_i * d_t_stage
        grad_terms = eqx.apply_updates(grad_terms, dterms)
        grad_args = eqx.apply_updates(grad_args, dargs)
        grad_control = eqx.apply_updates(grad_control, dcontrol)

    _, pullback = eqx.filter_vjp(
        ft.partial(_first_stage, b0=recurrence.B[0]), terms, t0, y0, args, control
    )
    dterms, d_t_stage, grad_y0, dargs, dcontrol = pullback((grad_y, grad_tmp))
    if t0_perturbed:
        grad_t0 = grad_t0 + d_t_stage
    grad_terms = eqx.apply_updates(grad_terms, dterms)
    grad_args = eqx.apply_updates(grad_args, dargs)
    grad_control = eqx.apply_updates(grad_control, dcontrol)

    if _any_perturbed(terms_perturbed):
        _, pullback = eqx.filter_vjp(lambda terms_: terms_.contr(t0, t1), terms)
        (dterms,) = pullback(grad_control)
        grad_terms = eqx.apply_updates(grad_terms, dterms)
    if t0_perturbed or t1_perturbed:
        _, pullback = eqx.filter_vjp(
            lambda t0_t1: terms.contr(t0_t1[0], t0_t1[1]), (t0, t1)
        )
        ((d_t0, d_t1),) = pullback(grad_control)
        if t0_perturbed:
            grad_t0 = grad_t0 + d_t0
        if t1_perturbed:
            grad_t1 = grad_t1 + d_t1
    grad_y0 = eqx.apply_updates(grad_y0, grad_dense_y0)

    return (
        grad_terms if _any_perturbed(terms_perturbed) else _none_tree(diff_terms),
        grad_t0 if t0_perturbed else None,
        grad_t1 if t1_perturbed else None,
        grad_y0,
        grad_args if _any_perturbed(args_perturbed) else _none_tree(diff_args),
    )


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
    def from_butcher(cls, tableau) -> "LowStorageRecurrence":
        """Construct from a :class:`diffrax.ButcherTableau`.

        Raises ``ValueError`` if the tableau does not have the structure required
        for a 2N Williamson representation.
        """
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

        # A[i-1] = (a_lower[i][i-1] - a_lower[i-1][i-1]) / B[i]  for i in 1..s-2
        # A[s-2]  = (b_sol[s-2] - B[s-2]) / B[s-1]
        A = np.empty(s - 1)
        for i in range(1, s - 1):
            A[i - 1] = (a_lower[i][i - 1] - a_lower[i - 1][i - 1]) / B[i]
        A[s - 2] = (b_sol[s - 2] - B[s - 2]) / B[s - 1]

        recurrence = cls(A=A, B=B, C=C)
        reconstructed = recurrence.to_butcher()

        if not np.allclose(reconstructed.b_sol, b_sol):
            raise ValueError(
                "Butcher tableau is not representable as a 2N low-storage method."
            )
        for r, given in zip(reconstructed.a_lower, a_lower):
            if not np.allclose(r, given):
                raise ValueError(
                    "Butcher tableau is not representable as a 2N low-storage method."
                )

        b_error = np.asarray(tableau.b_error)
        b_penultimate = np.append(reconstructed.a_lower[-1], 0.0)
        penultimate_stage_error = np.allclose(b_error, b_sol - b_penultimate)

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
