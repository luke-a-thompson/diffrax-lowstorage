from typing import ClassVar

import numpy as np

from .base import LowStorageRecurrence, LowStorageSolver

_ndbrk134_recurrence = LowStorageRecurrence(
    A=np.array(
        [
            -0.6160178650170565,
            -0.4449487060774118,
            -1.0952033345276178,
            -1.2256030785959187,
            -0.2740182222332805,
            -0.0411952089052647,
            -0.179708489915356,
            -1.1771530652064288,
            -0.4078831463120878,
            -0.8295636426191777,
            -4.7895970584252288,
            -0.6606671432964504,
        ]
    ),
    B=np.array(
        [
            0.0271990297818803,
            0.1772488819905108,
            0.0378528418949694,
            0.6086431830142991,
            0.21543139743161,
            0.2066152563885843,
            0.0415864076069797,
            0.0219891884310925,
            0.9893081222650993,
            0.0063199019859826,
            0.3749640721105318,
            1.6080235151003195,
            0.0961209123818189,
        ]
    ),
    C=np.array(
        [
            0.0,
            0.0271990297818803,
            0.0952594339119365,
            0.1266450286591127,
            0.1825883045699772,
            0.3737511439063931,
            0.5301279418422206,
            0.5704177433952291,
            0.5885784947099155,
            0.6160769826246714,
            0.6223252334314046,
            0.6897593128753419,
            0.9126827615920843,
        ]
    ),
)


class NDBRK134(LowStorageSolver):
    """13-stage, 4th-order 2N RK method by Niegemann, Diehl, and Busch.

    Corresponds to: NDBLSRK134 in Diffeq.jl.

    ??? Reference

        ```bibtex
        @article{Niegemann2012,
          title = {Efficient low-storage Runge–Kutta schemes with optimized stability regions},
          volume = {231},
          ISSN = {0021-9991},
          url = {http://dx.doi.org/10.1016/j.jcp.2011.09.003},
          DOI = {10.1016/j.jcp.2011.09.003},
          number = {2},
          journal = {Journal of Computational Physics},
          publisher = {Elsevier BV},
          author = {Niegemann,  Jens and Diehl,  Richard and Busch,  Kurt},
          year = {2012},
          month = Jan,
          pages = {364–372}
        }
        ```
    """

    recurrence: ClassVar[LowStorageRecurrence] = _ndbrk134_recurrence

    def order(self, terms):
        del terms
        return 4
