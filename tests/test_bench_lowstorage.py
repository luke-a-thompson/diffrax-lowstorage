import sys
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import pytest
from bwrrk33 import BWRRK33
from diffrax._solver.bosh3 import Bosh3

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "diffrax-lowstorage"
sys.path.insert(0, str(_SRC))


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
            throw=False,
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
def test_lowstorage_vs_bosh3_compiled_memory(problem_size):
    y0 = jnp.ones((problem_size,), dtype=jnp.float32)

    bwrrk33_total, bwrrk33_stats = _compiled_memory_bytes(BWRRK33, y0)
    bosh3_total, bosh3_stats = _compiled_memory_bytes(Bosh3, y0)

    ratio = bwrrk33_total / bosh3_total if bosh3_total else float("inf")
    print(
        "\ncompiled-memory-bytes "
        f"(size={problem_size}): "
        f"bwrrk33={bwrrk33_total}, "
        f"bosh3={bosh3_total}, "
        f"ratio={ratio:.6f}"
    )
    print(
        "bwrrk33 breakdown: "
        f"temp={bwrrk33_stats.temp_size_in_bytes}, "
        f"args={bwrrk33_stats.argument_size_in_bytes}, "
        f"out={bwrrk33_stats.output_size_in_bytes}, "
        f"alias={bwrrk33_stats.alias_size_in_bytes}"
    )
    print(
        "bosh3 breakdown: "
        f"temp={bosh3_stats.temp_size_in_bytes}, "
        f"args={bosh3_stats.argument_size_in_bytes}, "
        f"out={bosh3_stats.output_size_in_bytes}, "
        f"alias={bosh3_stats.alias_size_in_bytes}"
    )

    assert bwrrk33_total > 0
    assert bosh3_total > 0
