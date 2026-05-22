from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_bwrrk33_tableau = LowStorageRecurrence(
    A=np.array([-0.6376944718422022, -1.3066477177371079]),
    B=np.array([0.45737999756938819, 0.92529641092092174, 0.39381359467507099]),
    # Stage times for non-autonomous problems, derived from the equivalent ERK tableau.
    C=np.array([0.0, 0.45737999756938819, 0.7926200024306075]),
)


class BWRRK33(LowStorageSolver):
    """3-stage, 3rd-order 2N low-storage RK method in Williamson form.

    ??? Reference

        ```bibtex
        @article{Williamson1980,
          title = {Low-storage Runge-Kutta schemes},
          volume = {35},
          ISSN = {0021-9991},
          url = {http://dx.doi.org/10.1016/0021-9991(80)90033-9},
          DOI = {10.1016/0021-9991(80)90033-9},
          number = {1},
          journal = {Journal of Computational Physics},
          publisher = {Elsevier BV},
          author = {Williamson,  J. H.},
          year = {1980},
          month = Mar,
          pages = {48–56}
        }
        ```
    """

    recurrence: ClassVar[LowStorageRecurrence] = _bwrrk33_tableau

    def order(self, terms):
        del terms
        return 3
