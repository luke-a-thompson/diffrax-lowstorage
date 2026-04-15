from __future__ import annotations

from typing import ClassVar, override

import numpy as np
from diffrax import RESULTS, AbstractReversibleSolver, AbstractTerm
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from jaxtyping import PyTree

from diffrax_lowstorage import LowStorageRecurrence, LowStorageSolver

_ees25_recurrence = LowStorageRecurrence(
    A=np.array([-7 / 15, -35 / 32]),
    B=np.array([1 / 3, 15 / 16, 2 / 5]),
    C=np.array([0.0, 1 / 3, 5 / 6]),
)

_SolverState = Y


class EES25(LowStorageSolver, AbstractReversibleSolver):
    """2N-EES(2,5;1/4) solver."""

    recurrence: ClassVar[LowStorageRecurrence] = _ees25_recurrence

    @override
    def order(self, terms):
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 5

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
