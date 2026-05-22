from __future__ import annotations

from typing import ClassVar, override

import numpy as np
from diffrax import (
    RESULTS,
    AbstractReversibleSolver,
    AbstractStratonovichSolver,
    AbstractTerm,
)
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from jaxtyping import PyTree

from diffrax_lowstorage import LowStorageRecurrence, LowStorageSolver

_ees25_recurrence = LowStorageRecurrence(
    A=np.array([-7 / 15, -35 / 32]),
    B=np.array([1 / 3, 15 / 16, 2 / 5]),
    C=np.array([0.0, 1 / 3, 5 / 6]),
)

_SolverState = Y


class EES25(LowStorageSolver, AbstractReversibleSolver, AbstractStratonovichSolver):
    """2N-EES(2,5;1/4) solver.

    O(1)-reversible and converges to the Stratonovich solution.

    ??? Reference

        ```bibtex
        @misc{Shmelev2026,
          title = {Explicit and Effectively Symmetric Schemes for Neural SDEs on Lie Groups},
          author = {Shmelev,  Daniil and Thompson,  Luke and Salvi,  Cristopher},
          year = {2026},
          eprint = {2509.20599},
          archivePrefix = {arXiv},
          primaryClass = {cs.LG},
          doi = {10.48550/arXiv.2509.20599},
          url = {https://arxiv.org/abs/2509.20599}
        }
        ```
    """

    recurrence: ClassVar[LowStorageRecurrence] = _ees25_recurrence

    @override
    def order(self, terms):
        del terms
        return 2

    def strong_order(self, terms):
        del terms
        return 0.5

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
