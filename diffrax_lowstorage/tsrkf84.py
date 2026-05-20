from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_tsrkf84_recurrence = LowStorageRecurrence(
    A=np.array(
        [
            -0.5534431294501569,
            0.0106598757020349,
            -0.5515812888932,
            -1.885790377558741,
            -5.701295742793264,
            2.113903965664793,
            -0.533957882667528,
        ]
    ),
    B=np.array(
        [
            0.0803793688273695,
            0.5388497458569843,
            0.0197497440903196,
            0.0991184129733997,
            0.7466920411064123,
            1.679584245618894,
            0.2433728067008188,
            0.1422730459001373,
        ]
    ),
    C=np.array(
        [
            0.0,
            0.0803793688273695,
            0.321006425033843,
            0.340850182660466,
            0.385036482428547,
            0.50400524775341,
            0.657897756116854,
            0.9484087623348481,
        ]
    ),
)


class TSRKF84(LowStorageSolver):
    """8-stage, 4th-order 2N RK-F method by Toulorge and Desmet.

    Corresponds to: DGLDDRK84_F in Diffeq.jl.
    """

    recurrence: ClassVar[LowStorageRecurrence] = _tsrkf84_recurrence

    def order(self, terms):
        del terms
        return 4
