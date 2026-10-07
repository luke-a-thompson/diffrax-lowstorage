"""Forward and round-trip ODE convergence for EES25, EES27, and EES29.

Adapted from /home/luke/georax/docs/examples/solver_convergence.py.
The test equation is y' = -y**3, y(0) = 1, with exact solution 1/sqrt(1 + 2t).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from _convergence_plot import plot_curve, set_method_title

from diffrax_lowstorage import EES25, EES27, EES29

T0 = 0.0
T1 = 1.0
Y0 = 1.0


def make_method_runner(solver):
    """Compile the forward solve and actual backward_step reconstruction."""
    term = diffrax.ODETerm(lambda t, y, args: -(y**3))

    @jax.jit
    def run(num_steps):
        dt = (T1 - T0) / num_steps
        y0 = jnp.asarray(Y0, dtype=jnp.float64)
        state = solver.init(term, T0, T0 + dt, y0, None)

        def forward(i, carry):
            y, state = carry
            t0 = T0 + i * dt
            t1 = T0 + (i + 1) * dt
            y, _, _, state, _ = solver.step(term, t0, t1, y, None, state, False)
            return y, state

        y1, state = jax.lax.fori_loop(0, num_steps, forward, (y0, state))

        def backward(j, carry):
            y, state = carry
            i = num_steps - j
            t0 = T0 + (i - 1) * dt
            t1 = T0 + i * dt
            previous_t = jnp.maximum(T0, t0 - dt)
            y, _, state, _ = solver.backward_step(
                term, t0, t1, y, None, (previous_t,), state, False
            )
            return y, state

        recovered, _ = jax.lax.fori_loop(0, num_steps, backward, (y1, state))
        return y1, recovered

    return run


def plot_grid(hs, output_dir):
    solvers = [("EES25", EES25()), ("EES27", EES27()), ("EES29", EES29())]
    exact = Y0 / np.sqrt(1 + 2 * Y0**2 * (T1 - T0))
    fig, axes = plt.subplots(len(solvers), 2, figsize=(10, 3.2 * len(solvers)))
    results = {
        "equation": "y' = -y^3",
        "t0": T0,
        "t1": T1,
        "y0": Y0,
        "h": hs.tolist(),
        "solvers": {},
    }

    for i, (name, solver) in enumerate(solvers):
        run = make_method_runner(solver)
        values = np.array([run(round((T1 - T0) / h)) for h in hs])
        forward_errors = np.abs(values[:, 0] - exact)
        backward_errors = np.abs(values[:, 1] - Y0)
        method_results = {}
        for j, (mode, errors, rate) in enumerate(
            (
                ("forward", forward_errors, solver.order(None)),
                ("roundtrip", backward_errors, solver.antisymmetric_order(None)),
            )
        ):
            log_errors = np.log10(np.maximum(errors, np.finfo(np.float64).eps))
            method_results[mode] = plot_curve(
                name, hs, log_errors, rate, axes[i, j], backward=bool(j)
            )
            method_results[mode]["errors"] = errors.tolist()
            set_method_title(
                axes[i, j], solver.antisymmetric_order(None), backward=bool(j)
            )
        results["solvers"][name] = method_results

    fig.suptitle(r"$\dot{y} = -y^3$, $y(0) = 1$")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "solver_convergence.png"
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    output_path.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-power", type=int, default=2)
    parser.add_argument("--max-power", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    args = parser.parse_args()
    if args.min_power < 0 or args.max_power < args.min_power:
        parser.error("Require 0 <= min-power <= max-power.")
    return args


def main():
    args = parse_args()
    hs = 2.0 ** -np.arange(args.min_power, args.max_power + 1, dtype=np.float64)
    output_path = plot_grid(hs, args.output_dir)
    print(f"Saved {output_path} (also .pdf and .json)")


if __name__ == "__main__":
    main()
