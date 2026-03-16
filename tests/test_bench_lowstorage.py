import time

import diffrax
import jax
import jax.numpy as jnp
import pytest
from diffrax import Bosh3, Heun

from diffrax_lowstorage import BWRRK33, BWRRK53

SOLVERS = [
    ("heun", Heun),
    ("bosh3", Bosh3),
    ("bwrrk33", BWRRK33),
    ("bwrrk53", BWRRK53),
]


def _to_total_bytes(memory_stats) -> int:
    return int(
        memory_stats.temp_size_in_bytes
        + memory_stats.argument_size_in_bytes
        + memory_stats.output_size_in_bytes
        - memory_stats.alias_size_in_bytes
    )


def _compiled_memory_bytes(solver_cls, y0):
    solver = solver_cls()
    term = diffrax.ODETerm(lambda t, y, args: -10.0 * y**3)
    saveat = diffrax.SaveAt(t1=True)

    def run(y_init):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=y_init,
            saveat=saveat,
            throw=True,
        )
        return sol.ys

    compiled = jax.jit(run).lower(y0).compile()
    if not hasattr(compiled, "memory_analysis"):
        pytest.skip("Compiled executable does not expose memory_analysis().")

    memory_stats = compiled.memory_analysis()
    if memory_stats is None:
        pytest.skip("memory_analysis() returned None on this backend.")
    return _to_total_bytes(memory_stats), memory_stats


@pytest.mark.parametrize("problem_size", [8192])
def test_solvers_compiled_memory(problem_size):
    y0 = jnp.ones((problem_size,), dtype=jnp.float32)

    results = []
    for solver_name, solver_cls in SOLVERS:
        total, _ = _compiled_memory_bytes(solver_cls, y0)
        results.append((solver_name, total))

    print(f"\ncompiled-memory-bytes (size={problem_size}):")
    for name, total in results:
        ratio = total / results[0][1] if results[0][1] else float("inf")
        print(f"  {name}: {total} bytes  (vs {results[0][0]}: {ratio:.3f}x)")

    for name, total in results:
        assert total > 0, f"{name} reported non-positive compiled-memory bytes."


def _runtime_seconds(solver_cls, y0, n_repeats=100):
    solver = solver_cls()
    term = diffrax.ODETerm(lambda t, y, args: -10.0 * y**3)
    saveat = diffrax.SaveAt(t1=True)

    @jax.jit
    def run(y_init):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=y_init,
            saveat=saveat,
            throw=True,
        )
        return sol.ys

    # Warmup
    jax.block_until_ready(run(y0))

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        jax.block_until_ready(run(y0))
    return (time.perf_counter() - t0) / n_repeats


@pytest.mark.parametrize("problem_size", [8192])
def test_solvers_runtime(problem_size):
    y0 = jnp.ones((problem_size,), dtype=jnp.float32)

    results = []
    for solver_name, solver_cls in SOLVERS:
        t = _runtime_seconds(solver_cls, y0)
        results.append((solver_name, t))

    print(f"\nruntime (size={problem_size}):")
    for name, t in results:
        ratio = t / results[0][1] if results[0][1] else float("inf")
        print(f"  {name}: {t * 1e3:.3f} ms  (vs {results[0][0]}: {ratio:.3f}x)")

    for name, t in results:
        assert t > 0, f"{name} reported non-positive runtime."
