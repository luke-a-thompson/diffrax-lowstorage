import diffrax
import jax.numpy as jnp
import pytest
from diffrax_lowstorage import BWRRK33, BWRRK53


SOLVERS = [("bwrrk33", BWRRK33), ("bwrrk53", BWRRK53)]


@pytest.mark.parametrize(("solver_name", "solver_cls"), SOLVERS)
def test_lowstorage_solver_fixed_step(solver_name, solver_cls):
    del solver_name
    term = diffrax.ODETerm(lambda t, y, args: -10 * y**3)
    solver = solver_cls()

    t0 = 0.0
    t1 = 1.0
    dt0 = 0.01
    y0 = 1.0

    out = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=diffrax.SaveAt(t1=True),
        throw=True,
    )

    expected = 1.0 / jnp.sqrt(1.0 + 20.0 * t1)
    assert out.result == diffrax.RESULTS.successful, "Result reported as unsuccessful"
    assert jnp.allclose(out.ys[0], expected, rtol=2e-2, atol=2e-3)
