from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_bprko52_recurrence = LowStorageRecurrence(
    A=np.array([-1.0, -1.55798, -1.0, -0.45031]),
    B=np.array([0.2, 0.83204, 0.6, 0.35394, 0.2]),
    C=np.array([0.0, 0.2, 0.2, 0.8, 0.8]),
)


class BPRKO52(LowStorageSolver):
    """5-stage, 2nd-order 2N low-storage optimized RK method.

    Corresponds to: ORK256 in Diffeq.jl.
    """

    recurrence: ClassVar[LowStorageRecurrence] = _bprko52_recurrence

    def order(self, terms):
        del terms
        return 2
