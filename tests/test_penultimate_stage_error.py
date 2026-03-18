import diffrax
import jax.numpy as jnp
import pytest

from conftest import NO_ERROR_SOLVERS, PENULTIMATE_ERROR_SOLVERS


@pytest.mark.parametrize(("solver_name", "solver_cls"), PENULTIMATE_ERROR_SOLVERS)
def test_penultimate_stage_error_supports_pid(solver_name, solver_cls):
    del solver_name
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = solver_cls()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-7)

    out = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=0.1,
        y0=1.0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=controller,
        throw=True,
    )

    expected = jnp.exp(-1.0)
    assert out.result == diffrax.RESULTS.successful
    assert out.ys is not None
    assert jnp.allclose(out.ys[0], expected, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize(("solver_name", "solver_cls"), NO_ERROR_SOLVERS)
def test_no_error_solver_not_supported(solver_name, solver_cls):
    del solver_name
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = solver_cls()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-7)

    with pytest.raises(RuntimeError, match="does not provide error estimates"):
        diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=0.1,
            dt0=0.01,
            y0=1.0,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=controller,
            throw=True,
        )
