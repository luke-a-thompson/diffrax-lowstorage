import sys
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import pytest
from bwrrk33 import BWRRK33
from diffrax import Bosh3, Heun

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "diffrax-lowstorage"
sys.path.insert(0, str(_SRC))

SOLVERS = [
    ("heun", Heun),
    ("bosh3", Bosh3),
    ("bwrrk33", BWRRK33),
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
        total, stats = _compiled_memory_bytes(solver_cls, y0)
        results.append((solver_name, total, stats))

    summary = ", ".join(f"{name}={total}" for name, total, _ in results)
    print(f"\ncompiled-memory-bytes (size={problem_size}): {summary}")
    for name, total, _ in results:
        ratio = total / results[0][1] if results[0][1] else float("inf")
        print(f"{name} vs {results[0][0]} ratio={ratio:.6f}")
    for name, _, stats in results:
        print(
            f"{name} breakdown: "
            f"temp={stats.temp_size_in_bytes}, "
            f"args={stats.argument_size_in_bytes}, "
            f"out={stats.output_size_in_bytes}, "
            f"alias={stats.alias_size_in_bytes}"
        )

    for name, total, _ in results:
        assert total > 0, f"{name} reported non-positive compiled-memory bytes."
