from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_tsrkc73_recurrence = LowStorageRecurrence(
    A=np.array(
        [
            -0.808316387498383,
            -1.503407858773331,
            -1.053064525050744,
            -1.463149119280508,
            -0.659288128108783,
            -1.667891931891068,
        ]
    ),
    B=np.array(
        [
            0.0119705267309784,
            0.8886897793820711,
            0.4578382089261419,
            0.5790045253338471,
            0.3160214638138484,
            0.2483525368264122,
            0.0677123095940884,
        ]
    ),
    C=np.array(
        [
            0.0,
            0.0119705267309784,
            0.182317794036199,
            0.5082168062551849,
            0.653203122014859,
            0.853440138567825,
            0.998046608462379,
        ]
    ),
)


class TSRKC73(LowStorageSolver):
    """7-stage, 3rd-order 2N RK-C method by Toulorge and Desmet.

    Corresponds to: DGLDDRK73_C in Diffeq.jl.

    ??? Reference

        ```bibtex
        @article{Toulorge2012,
          title = {Optimal Runge–Kutta schemes for discontinuous Galerkin space discretizations applied to wave propagation problems},
          volume = {231},
          ISSN = {0021-9991},
          url = {http://dx.doi.org/10.1016/j.jcp.2011.11.024},
          DOI = {10.1016/j.jcp.2011.11.024},
          number = {4},
          journal = {Journal of Computational Physics},
          publisher = {Elsevier BV},
          author = {Toulorge,  T. and Desmet,  W.},
          year = {2012},
          month = Feb,
          pages = {2067–2091}
        }
        ```
    """

    recurrence: ClassVar[LowStorageRecurrence] = _tsrkc73_recurrence

    def order(self, terms):
        del terms
        return 3
