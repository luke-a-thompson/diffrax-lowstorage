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


def _measure_order(term, solver, t0, t1, y0, exact, dts):
    log_dts = jnp.log(jnp.abs(dts))
    errors = []
    for dt in dts:
        out = diffrax.diffeqsolve(
            term, solver, t0, t1, float(dt), y0, saveat=diffrax.SaveAt(t1=True)
        )
        assert out.ys is not None
        errors.append(float(jnp.abs(out.ys[0] - exact)))
    log_errs = jnp.log(jnp.array(errors))
    return float(jnp.polyfit(log_dts, log_errs, 1)[0])


@pytest.mark.parametrize(("solver_name", "solver_cls"), SOLVERS)
def test_lowstorage_solver_order(solver_name, solver_cls):
    term = diffrax.ODETerm(lambda t, y, args: -10 * y**3)
    solver = solver_cls()
    expected_order = solver.order(term)

    t0, t1, y0 = 0.0, 1.0, 1.0
    exact_t1 = 1.0 / jnp.sqrt(1.0 + 20.0 * t1)
    dts = jnp.array([0.025, 0.0125, 0.00625, 0.003125])

    slope = _measure_order(term, solver, t0, t1, y0, exact_t1, dts)
    print(f"\n{solver_name} forward:  expected={expected_order}, measured={slope:.2f}")
    assert slope >= expected_order * 0.9, (
        f"Expected order ~{expected_order}, got slope {slope:.2f}"
    )

    if isinstance(solver, diffrax.AbstractReversibleSolver):
        expected_rt_order = solver.antisymmetric_order(term)
        # Larger steps than the forward test: high-order cancellation is visible
        # before floating-point noise dominates.
        dts_rt = jnp.array([1 / 64, 1 / 128, 1 / 256, 1 / 512])
        rt_errors = []
        for dt in dts_rt:
            out_fwd = diffrax.diffeqsolve(
                term, solver, t0, t1, float(dt), y0, saveat=diffrax.SaveAt(t1=True)
            )
            assert out_fwd.ys is not None
            out_bwd = diffrax.diffeqsolve(
                term, solver, t1, t0, -float(dt), float(out_fwd.ys[0]), saveat=diffrax.SaveAt(t1=True)
            )
            assert out_bwd.ys is not None
            rt_errors.append(float(jnp.abs(out_bwd.ys[0] - y0)))
        slope_rt = float(jnp.polyfit(jnp.log(dts_rt), jnp.log(jnp.array(rt_errors)), 1)[0])
        print(f"{solver_name} roundtrip: expected={expected_rt_order}, measured={slope_rt:.2f}")
        assert slope_rt >= expected_rt_order * 0.9, (
            f"Round-trip: expected order ~{expected_rt_order}, got slope {slope_rt:.2f}"
        )
