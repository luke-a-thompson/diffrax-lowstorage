from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_tsrkc84_recurrence = LowStorageRecurrence(
    A=np.array(
        [
            -0.721296248227924,
            -0.0107733657161298,
            -0.516258469893097,
            -1.730100286632201,
            -5.200129304403076,
            0.783705894541642,
            -0.544583609433219,
        ]
    ),
    B=np.array(
        [
            0.2165936736758085,
            0.1773950826411583,
            0.0180253861162329,
            0.0847347637254149,
            0.8129106974622483,
            1.90341603042276,
            0.1314841743399048,
            0.2082583170674149,
        ]
    ),
    C=np.array(
        [
            0.0,
            0.2165936736758085,
            0.266034348753817,
            0.284005612252272,
            0.325126684378857,
            0.455514959918753,
            0.771321931710117,
            0.919902896453866,
        ]
    ),
)


class TSRKC84(LowStorageSolver):
    """8-stage, 4th-order 2N RK-C method by Toulorge and Desmet.

    Corresponds to: DGLDDRK84_C in Diffeq.jl.
    """

    recurrence: ClassVar[LowStorageRecurrence] = _tsrkc84_recurrence

    def order(self, terms):
        del terms
        return 4
