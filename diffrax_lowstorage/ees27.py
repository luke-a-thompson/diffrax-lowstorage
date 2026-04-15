from __future__ import annotations

from typing import ClassVar, override

import numpy as np
from diffrax import AbstractReversibleSolver

from diffrax_lowstorage import LowStorageRecurrence, LowStorageSolver

_ees27_recurrence = LowStorageRecurrence(
    A=np.array([1.0 - np.sqrt(2.0), -1.0, -(1.0 + np.sqrt(2.0))]),
    B=np.array(
        [
            0.5 * (2.0 - np.sqrt(2.0)),
            0.5 * np.sqrt(2.0),
            0.5 * np.sqrt(2.0),
            0.25 * (2.0 - np.sqrt(2.0)),
        ]
    ),
    C=np.array(
        [
            0.0,
            0.5 * (2.0 - np.sqrt(2.0)),
            0.5 * np.sqrt(2.0),
            1.0,
        ]
    ),
)


class EES27(LowStorageSolver, AbstractReversibleSolver):
    """Commutator-free EES(2,5;1/4) solver with chained exponentials."""

    recurrence: ClassVar[LowStorageRecurrence] = _ees27_recurrence

    @override
    def order(self, terms):
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 7
