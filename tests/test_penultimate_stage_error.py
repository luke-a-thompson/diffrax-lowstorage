import diffrax
import jax.numpy as jnp
import pytest
from diffrax_lowstorage import BWRRK33, BWRRK53


def test_bwrrk53_penultimate_stage_error_supports_pid():
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = BWRRK53()
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
    assert jnp.allclose(out.ys[0], expected, rtol=2e-3, atol=2e-4)


def test_bwrrk33_penultimate_stage_error_not_supported():
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = BWRRK33()
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
