from typing import ClassVar

import numpy as np

from diffrax_lowstorage.base import LowStorageRecurrence, LowStorageSolver

_bwrrk53_tableau = LowStorageRecurrence(
    # Williamson A-form coefficients (A1 is implicitly 0, so we store A2..A5)
    A=np.array([-5.0 / 8.0, -4.0 / 3.0, -3.0 / 4.0, -8.0 / 5.0]),
    # Williamson B coefficients (B1..B5)
    B=np.array([1.0 / 4.0, 2.0 / 3.0, 1.0 / 2.0, 2.0 / 5.0, 1.0 / 9.0]),
    # Stage times for non-autonomous problems (from the equivalent ERK tableau)
    C=np.array([0.0, 1.0 / 4.0, 1.0 / 2.0, 3.0 / 4.0, 1.0]),
    penultimate_stage_error=True,
)


class BWRRK53(LowStorageSolver):
    """5-stage, 3rd-order Williamson 2N low-storage RK method.
    Penultimate stage (the state before the final update) is 2nd-order accurate at t+h,
    so you get an embedded (3,2) pair with error ≈ y_final - y_penultimate.
    """

    tableau: ClassVar[LowStorageRecurrence] = _bwrrk53_tableau

    def order(self, terms):
        del terms
        return 3
