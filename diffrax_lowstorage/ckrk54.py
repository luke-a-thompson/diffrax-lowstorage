from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_tableau = LowStorageRecurrence(
    # Williamson A-form coefficients (A1 is implicitly 0, so we store A2..A5)
    A=np.array([-0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]),
    # Williamson B coefficients (B1..B5)
    B=np.array(
        [
            0.1496590219993,
            0.3792103129999,
            0.8229550293869,
            0.6994504559488,
            0.1530572479681,
        ]
    ),
    # Stage times for non-autonomous problems (from the equivalent ERK tableau)
    C=np.array(
        [0.0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748]
    ),
)


class CKRK54(LowStorageSolver):
    """5-stage, 4td-order Williamson 2N low-storage RK method.

    Reference:
        Carpenter, Mark H., and Christopher A. Kennedy. 1994. Fourth-Order 2N-Storage Runge-Kutta Schemes. NASA-TM-109112. https://ntrs.nasa.gov/citations/19940028444.
        Table 3, Solution 3
    """

    tableau: ClassVar[LowStorageRecurrence] = _tableau

    def order(self, terms):
        del terms
        return 4
