from __future__ import annotations

from typing import ClassVar, override

import numpy as np
from diffrax import AbstractReversibleSolver

from diffrax_lowstorage import LowStorageRecurrence, LowStorageSolver

_ees25_recurrence = LowStorageRecurrence(
    A=np.array([-0.5, -2.0]),
    B=np.array([0.5, 1.0, 0.25]),
    C=np.array([0.0, 0.5, 1.0]),
)


class EES25(LowStorageSolver, AbstractReversibleSolver):
    """Commutator-free EES(2,5;1/4) solver with chained exponentials."""

    recurrence: ClassVar[LowStorageRecurrence] = _ees25_recurrence

    @override
    def order(self, terms):
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 5
