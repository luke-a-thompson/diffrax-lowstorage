from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import RESULTS, AbstractSolver, AbstractTerm, LocalLinearInterpolation

from .base import LowStorageRecurrence, _low_storage_step

_shrk2n_first_recurrence = LowStorageRecurrence(
    A=np.array([-0.6051226, -2.0437564, -0.7406999, -4.4231765]),
    B=np.array([0.2687454, 0.8014706, 0.505157, 0.5623568, 0.0590065]),
    C=np.array([0.0, 0.2687454, 0.585228, 0.6827066, 1.1646854]),
)

_shrk2n_second_recurrence = LowStorageRecurrence(
    A=np.array([-0.4412737, -1.073982, -1.706357, -2.7979293, -4.0913537]),
    B=np.array([0.1158488, 0.3728769, 0.7379536, 0.579811, 1.0312849, 0.15]),
    C=np.array([0.0, 0.1158485, 0.324185, 0.6193208, 0.8034472, 0.9184166]),
)


class SHRK2N(AbstractSolver):
    """Alternating 2N RK method by Stanescu and Habashi.

    Corresponds to: SHLDDRK_2N in Diffeq.jl.

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

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar = LocalLinearInterpolation

    def order(self, terms):
        del terms
        return 4

    def error_order(self, terms):
        del terms
        return None

    def init(
        self,
        terms: AbstractTerm,
        t0,
        t1,
        y0,
        args,
    ):
        del terms, t0, t1, y0, args
        return jnp.asarray(False)

    def step(
        self,
        terms: AbstractTerm,
        t0,
        t1,
        y0,
        args,
        solver_state,
        made_jump,
    ):
        def first_step(_):
            return _low_storage_step(
                (terms, t0, t1, y0, args),
                recurrence=_shrk2n_first_recurrence,
            )

        def second_step(_):
            return _low_storage_step(
                (terms, t0, t1, y0, args),
                recurrence=_shrk2n_second_recurrence,
            )

        use_second = jnp.logical_and(solver_state, jnp.logical_not(made_jump))
        y1, y_error, dense_info = jax.lax.cond(
            use_second,
            second_step,
            first_step,
            operand=None,
        )
        return (
            y1,
            y_error,
            dense_info,
            jnp.logical_not(use_second),
            RESULTS.successful,
        )

    def func(
        self,
        terms: AbstractTerm,
        t0,
        y0,
        args,
    ):
        return terms.vf(t0, y0, args)
