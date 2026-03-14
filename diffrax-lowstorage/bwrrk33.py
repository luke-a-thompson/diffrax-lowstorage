import jax.numpy as jnp

from base import TwoNSolver


class BWRRK33(TwoNSolver):
    """3-stage, 3rd-order 2N low-storage RK method in Williamson form."""

    def __init__(self):
        super().__init__(
            A=jnp.array(
                [
                    -0.6376944718422022,
                    -1.3066477177371079,
                ]
            ),
            B=jnp.array(
                [
                    0.45737999756938819,
                    0.92529641092092174,
                    0.39381359467507099,
                ]
            ),
            # Stage times for non-autonomous problems, derived from the equivalent ERK tableau.
            C=jnp.array(
                [
                    0.0,
                    0.45737999756938819,
                    0.7926200024306075,
                ]
            ),
        )
