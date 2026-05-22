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

    recurrence: ClassVar[LowStorageRecurrence] = _tsrkf84_recurrence

    def order(self, terms):
        del terms
        return 4
