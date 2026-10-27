import diffrax
import jax.numpy as jnp
import pytest
from conftest import NO_ERROR_SOLVERS, PENULTIMATE_ERROR_SOLVERS

from diffrax_lowstorage import BWRRK53


def test_error_estimate_survives_rounded_away_state_updates():
    # For y' = t^2 on [0, 1], the two formulas add 1/3 and 3/10.
    # Both increments round away at this state magnitude; their error must not.
    term = diffrax.ODETerm(lambda t, y, args: jnp.full_like(y, t**2))
    y0 = jnp.array(1e20)
    y1, error, _, _, _ = BWRRK53().step(term, 0.0, 1.0, y0, None, None, False)
    assert y1 == y0
    assert error is not None
    assert jnp.allclose(error, 1 / 30, rtol=1e-6, atol=0)


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
