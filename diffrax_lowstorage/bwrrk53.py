import jax.numpy as jnp

from .base import LowStorageSolver


class BWRRK53(LowStorageSolver):
    """5-stage, 3rd-order Williamson 2N low-storage RK method.
    Penultimate stage (the state before the final update) is 2nd-order accurate at t+h,
    so you get an embedded (3,2) pair with error ≈ y_final - y_penultimate.
    """

    def order(self, terms):
        del terms
        return 3

    def __init__(self):
        super().__init__(
            # Williamson A-form coefficients (A1 is implicitly 0, so we store A2..A5)
            A=jnp.array(
                [
                    -5.0 / 8.0,  # A2
                    -4.0 / 3.0,  # A3
                    -3.0 / 4.0,  # A4
                    -8.0 / 5.0,  # A5
                ]
            ),
            # Williamson B coefficients (B1..B5)
            B=jnp.array(
                [
                    1.0 / 4.0,  # B1
                    2.0 / 3.0,  # B2
                    1.0 / 2.0,  # B3
                    2.0 / 5.0,  # B4
                    1.0 / 9.0,  # B5
                ]
            ),
            # Stage times for non-autonomous problems (from the equivalent ERK tableau)
            C=jnp.array(
                [
                    0.0,
                    1.0 / 4.0,
                    1.0 / 2.0,
                    3.0 / 4.0,
                    1.0,
                ]
            ),
            use_penultimate_stage_error=True,
        )
