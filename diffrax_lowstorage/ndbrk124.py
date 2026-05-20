from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_ndbrk124_recurrence = LowStorageRecurrence(
    A=np.array(
        [
            -0.0923311242368072,
            -0.9441056581158819,
            -4.3271273247576394,
            -2.1557771329026072,
            -0.9770727190189062,
            -0.7581835342571139,
            -1.7977525470825499,
            -2.691566797270077,
            -4.6466798960268143,
            -0.1539613783825189,
            -0.5943293901830616,
        ]
    ),
    B=np.array(
        [
            0.0650008435125904,
            0.0161459902249842,
            0.5758627178358159,
            0.1649758848361671,
            0.3934619494248182,
            0.0443509641602719,
            0.2074504268408778,
            0.6914247433015102,
            0.3766646883450449,
            0.0757190350155483,
            0.2027862031054088,
            0.2167029365631842,
        ]
    ),
    C=np.array(
        [
            0.0,
            0.0650008435125904,
            0.0796560563081853,
            0.1620416710085376,
            0.2248877362907778,
            0.2952293985641261,
            0.3318332506149405,
            0.4094724050198658,
            0.6356954475753369,
            0.6806551557645497,
            0.714377371241835,
            0.9032588871651854,
        ]
    ),
)


class NDBRK124(LowStorageSolver):
    """12-stage, 4th-order 2N RK method by Niegemann, Diehl, and Busch.

    Corresponds to: NDBLSRK124 in Diffeq.jl.
    """

    recurrence: ClassVar[LowStorageRecurrence] = _ndbrk124_recurrence

    def order(self, terms):
        del terms
        return 4
