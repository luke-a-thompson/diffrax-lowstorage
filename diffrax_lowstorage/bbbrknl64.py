from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_bbbrknl64_recurrence = LowStorageRecurrence(
    A=np.array(
        [
            -0.737101392796,
            -1.634740794343,
            -0.74473900378,
            -1.469897351522,
            -2.813971388035,
        ]
    ),
    B=np.array(
        [
            0.032918605146,
            0.8232569982,
            0.3815309489,
            0.200092213184,
            1.718581042715,
            0.27,
        ]
    ),
    C=np.array(
        [
            0.0,
            0.032918605146,
            0.249351723343,
            0.466911705055,
            0.582030414044,
            0.847252983783,
        ]
    ),
)


class BBBRKNL64(LowStorageSolver):
    """6-stage, 4th-order nonlinear 2N RK method by Berland, Bogey, and Bailly.

    Corresponds to: RK46NL in Diffeq.jl.
    """

    recurrence: ClassVar[LowStorageRecurrence] = _bbbrknl64_recurrence

    def order(self, terms):
        del terms
        return 4
