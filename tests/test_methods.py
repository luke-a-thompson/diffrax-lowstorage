import diffrax
import jax
import jax.numpy as jnp
import pytest
from conftest import SOLVERS

# Needed to get slope for high order methods
jax.config.update("jax_enable_x64", True)


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
    assert out.ys is not None
    assert jnp.allclose(out.ys[0], expected, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize(("solver_name", "solver_cls"), SOLVERS)
def test_lowstorage_solver_order(solver_name, solver_cls):
    del solver_name
    term = diffrax.ODETerm(lambda t, y, args: -10 * y**3)
    solver = solver_cls()
    expected_order = solver.order(term)

    t0, t1, y0 = 0.0, 1.0, 1.0
    exact = 1.0 / jnp.sqrt(1.0 + 20.0 * t1)

    dts = jnp.array([0.1, 0.05, 0.025, 0.0125])
    errors = []
    for dt in dts:
        out = diffrax.diffeqsolve(
            term, solver, t0, t1, float(dt), y0, saveat=diffrax.SaveAt(t1=True)
        )
        assert out.ys is not None
        errors.append(float(jnp.abs(out.ys[0] - exact)))

    log_dts = jnp.log(dts)
    log_errs = jnp.log(jnp.array(errors))
    slope = float(jnp.polyfit(log_dts, log_errs, 1)[0])

    assert slope >= expected_order * 0.9, (
        f"Expected order ~{expected_order}, got slope {slope:.2f}"
    )
