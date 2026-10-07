from __future__ import annotations

from typing import ClassVar

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

from diffrax_lowstorage import LowStorageRecurrence, LowStorageSolver

_s5 = np.sqrt(5.0)

_ees29_recurrence = LowStorageRecurrence(
    A=np.array([(_s5 - 3) / 2, -(1 + _s5) / 4, 1 - _s5, -(3 + _s5) / 2]),
    B=np.array([(3 - _s5) / 4, 1 / 2, (_s5 - 1) / 2, 1 / 2, (3 - _s5) / 8]),
    C=np.array([0.0, (3 - _s5) / 4, 1 / 2, (1 + _s5) / 4, 1.0]),
)

_SolverState = Y


class EES29(LowStorageSolver, AbstractReversibleSolver, AbstractStratonovichSolver):
    """Five-stage 2N-EES(2,9) solver.

    Approximately reversible and converges to the Stratonovich solution.
    """

    recurrence: ClassVar[LowStorageRecurrence] = _ees29_recurrence

    @override
    def order(self, terms):
        del terms
        return 2

    def strong_order(self, terms):
        del terms
        return 0.5

    def antisymmetric_order(self, terms):
        del terms
        return 9

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
