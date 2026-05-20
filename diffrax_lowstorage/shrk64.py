from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_shrk64_recurrence = LowStorageRecurrence(
    A=np.array([-0.4919575, -0.8946264, -1.5526678, -3.4077973, -1.074264]),
    B=np.array([0.1453095, 0.4653797, 0.4675397, 0.7795279, 0.3574327, 0.15]),
    C=np.array([0.0, 0.1453095, 0.3817422, 0.6367813, 0.7560744, 0.9271047]),
)


class SHRK64(LowStorageSolver):
    """6-stage, 4th-order 2N low-storage RK method by Stanescu and Habashi.

    Corresponds to: SHLDDRK64 in Diffeq.jl.
    """

    recurrence: ClassVar[LowStorageRecurrence] = _shrk64_recurrence

    def order(self, terms):
        del terms
        return 4
