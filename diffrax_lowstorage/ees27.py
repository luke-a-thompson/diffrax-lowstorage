from __future__ import annotations

from typing import ClassVar, override

import numpy as np
from diffrax import AbstractReversibleSolver, AbstractTerm, RESULTS
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from jaxtyping import PyTree

from diffrax_lowstorage import LowStorageRecurrence, LowStorageSolver

_s2 = np.sqrt(2.0)

_ees27_recurrence = LowStorageRecurrence(
    A=np.array([(-7 + 4 * _s2) / 3, -(4 + 5 * _s2) / 12, 3 * (-31 + 8 * _s2) / 49]),
    B=np.array([(2 - _s2) / 3, (4 + _s2) / 8, 3 * (3 - _s2) / 7, (9 - 4 * _s2) / 14]),
    C=np.array([0.0, (2 - _s2) / 3, (2 + _s2) / 6, (4 + _s2) / 6]),
)
_SolverState = Y


class EES27(LowStorageSolver, AbstractReversibleSolver):
    """2N-EES(2,7;(1/4)) solver."""

    recurrence: ClassVar[LowStorageRecurrence] = _ees27_recurrence

    @override
    def order(self, terms):
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 7

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
        y0, _, dense_info, solver_state, result = self.step(
            terms, t1, t0, y1, args, solver_state, made_jump
        )
        return y0, dense_info, solver_state, result
