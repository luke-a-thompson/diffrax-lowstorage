from __future__ import annotations

import equinox as eqx
import numpy as np
from diffrax import (
    RESULTS,
    AbstractReversibleSolver,
    AbstractStratonovichSolver,
    AbstractTerm,
)
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from jaxtyping import PyTree
from typing_extensions import override

from .base import LowStorageRecurrence, LowStorageSolver, _SolverState


class CosineEES(LowStorageSolver, AbstractReversibleSolver, AbstractStratonovichSolver):
    """Second-order cosine EES method in 2N storage.

    `reversible_order` must be an odd integer >= 3; the method uses
    `(reversible_order + 1) // 2` stages. Approximately reversible and converges
    to the Stratonovich solution with strong order 0.5.
    """

    reversible_order: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        p = self.reversible_order
        if not isinstance(p, (int, np.integer)) or p < 3 or p % 2 != 1:
            raise ValueError("reversible_order must be an odd integer >= 3.")

    @property
    def recurrence(self) -> LowStorageRecurrence:
        s = (self.reversible_order + 1) // 2
        theta = np.pi / s
        angles = (np.arange(s) + 0.5) * theta
        # d_i = cos((i - 1) * theta) - cos(i * theta), without cancellation.
        d = 2 * np.sin(theta / 2) * np.sin(angles)
        b = d.copy()
        b[-1] /= 2
        c = 0.5 - np.cos(angles) / (2 * np.cos(theta / 2))
        c[0], c[-1] = 0.0, 1.0
        return LowStorageRecurrence(A=-d[:-1] / d[1:], B=b, C=c)

    @override
    def order(self, terms: PyTree[AbstractTerm]) -> int:
        del terms
        return 2

    def strong_order(self, terms: PyTree[AbstractTerm]) -> float:
        del terms
        return 0.5

    def antisymmetric_order(self, terms: PyTree[AbstractTerm]) -> int:
        del terms
        return self.reversible_order

    @override
    def backward_step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y1: Y,
        args: Args,
        ts_state: PyTree[RealScalarLike],
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, DenseInfo, _SolverState, RESULTS]:
        y0, _, _, solver_state, result = self.step(
            terms, t1, t0, y1, args, solver_state, made_jump
        )
        return y0, {"y0": y0, "y1": y1}, solver_state, result
