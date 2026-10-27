import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffrax_lowstorage import EES25, EES27, CosineEES


@pytest.mark.parametrize("solver", [EES25(), EES27(), CosineEES(9), CosineEES(11)])
def test_backward_step_interpolation_has_forward_orientation(solver):
    term = diffrax.ODETerm(lambda t, y, args: jnp.ones_like(y))
    y0 = jnp.array([0.0, 2.0])
    y1 = solver.step(term, 0.0, 1.0, y0, None, None, False)[0]
    recovered, dense_info, _, _ = solver.backward_step(
        term, 0.0, 1.0, y1, None, (), None, False
    )
    np.testing.assert_allclose(recovered, y0, atol=1e-6)
    interpolator = solver.interpolation_cls(t0=0.0, t1=1.0, **dense_info)
    for t in (0.0, 0.25, 1.0):
        np.testing.assert_allclose(interpolator.evaluate(t), y0 + t, atol=1e-6)


def test_cosine_ees3_matches_heun_under_jit():
    term = diffrax.ODETerm(lambda t, y, args: -t * y**2)
    y0 = jnp.array([0.2, 0.5])

    @jax.jit
    def step(solver, y):
        return solver.step(term, 0.1, 0.3, y, None, None, False)[0]

    np.testing.assert_allclose(step(CosineEES(3), y0), step(diffrax.Heun(), y0))
