from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_shrk52_recurrence = LowStorageRecurrence(
    A=np.array([-0.6913065, -2.655155, -0.8147688, -0.6686587]),
    B=np.array([0.1, 0.75, 0.7, 0.479313, 0.310392]),
    C=np.array([0.0, 0.1, 0.3315201, 0.4577796, 0.8666528]),
)


class SHRK52(LowStorageSolver):
    """5-stage, 2nd-order 2N low-storage RK method by Stanescu and Habashi.

    Corresponds to: SHLDDRK52 in Diffeq.jl.

    ??? Reference

        ```bibtex
        @article{Stanescu1998,
          title = {2N-Storage Low Dissipation and Dispersion Runge-Kutta Schemes for Computational Acoustics},
          volume = {143},
          ISSN = {0021-9991},
          url = {http://dx.doi.org/10.1006/jcph.1998.5986},
          DOI = {10.1006/jcph.1998.5986},
          number = {2},
          journal = {Journal of Computational Physics},
          publisher = {Elsevier BV},
          author = {Stanescu,  D. and Habashi,  W. G.},
          year = {1998},
          month = Jul,
          pages = {674–681}
        }
        ```
    """

    recurrence: ClassVar[LowStorageRecurrence] = _shrk52_recurrence

    def order(self, terms):
        del terms
        return 2
