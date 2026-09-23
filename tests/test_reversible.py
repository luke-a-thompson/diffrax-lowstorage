import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

from diffrax_lowstorage import EES25, EES27


@pytest.mark.parametrize("solver_cls", [EES25, EES27])
def test_backward_step_interpolation_has_forward_orientation(solver_cls):
    solver = solver_cls()
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
