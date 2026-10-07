"""Decimal convergence experiment for EES25, EES27, and EES29.

All coefficients, square roots, solution values, and errors use Decimal arithmetic
at 80 digits by default. Only computed log-errors are converted to float for
plotting and slope fitting. Existing float64 scripts and outputs are untouched.

Run: python docs/examples/solver_convergence_decimal.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.markers as mkr
import matplotlib.pyplot as plt
from _convergence_plot import (
    XLABEL,
    YLABEL_BACKWARD,
    YLABEL_FORWARD,
    set_method_title,
)


@dataclass(frozen=True)
class DecimalRecurrence:
    name: str
    antisymmetric_order: int
    A: tuple[Decimal, ...]
    B: tuple[Decimal, ...]


def make_recurrences():
    """Recompute the exact coefficient formulas in the active Decimal context.

    Stage times C are unnecessary for the autonomous equation y' = -y**3.
    """
    d = Decimal
    s2 = d(2).sqrt()
    s5 = d(5).sqrt()
    return (
        DecimalRecurrence(
            "EES25",
            5,
            A=(-d(7) / 15, -d(35) / 32),
            B=(d(1) / 3, d(15) / 16, d(2) / 5),
        ),
        DecimalRecurrence(
            "EES27",
            7,
            A=((-7 + 4 * s2) / 3, -(4 + 5 * s2) / 12, 3 * (-31 + 8 * s2) / 49),
            B=((2 - s2) / 3, (4 + s2) / 8, 3 * (3 - s2) / 7, (9 - 4 * s2) / 14),
        ),
        DecimalRecurrence(
            "EES29",
            9,
            A=((s5 - 3) / 2, -(1 + s5) / 4, 1 - s5, -(3 + s5) / 2),
            B=((3 - s5) / 4, d(1) / 2, (s5 - 1) / 2, d(1) / 2, (3 - s5) / 8),
        ),
    )


def step(recurrence, y, h):
    """The same two-register Williamson recurrence, evaluated with Decimal."""
    tmp = Decimal(0)
    for i, b in enumerate(recurrence.B):
        k = -h * y**3
        tmp = recurrence.A[i - 1] * tmp + k if i else k
        y += b * tmp
    return y


def solve_roundtrip(recurrence, num_steps):
    h = Decimal(1) / num_steps
    y = Decimal(1)
    for _ in range(num_steps):
        y = step(recurrence, y, h)
    forward_y = y
    for _ in range(num_steps):
        y = step(recurrence, y, -h)
    return forward_y, y


def plot_curve(name, hs, errors, expected_slope, precision, ax, *, backward=False):
    """Use the reference chart style with a Decimal precision cutoff."""
    unit_roundoff = Decimal(10) ** (1 - precision)
    roundoff = [100 * unit_roundoff / h for h in hs]
    resolved = np.array([error > cutoff for error, cutoff in zip(errors, roundoff)])
    x = np.array([float(h.log10()) for h in hs])
    y = np.array([float(max(error, unit_roundoff).log10()) for error in errors])
    measured = (
        float(np.polyfit(x[resolved], y[resolved], 1)[0])
        if resolved.sum() >= 2
        else None
    )
    err_label = YLABEL_BACKWARD if backward else YLABEL_FORWARD
    ax.scatter(
        x[resolved],
        y[resolved],
        marker=mkr.MarkerStyle("x", fillstyle="none"),
        color="crimson",
        label=err_label,
    )
    if resolved.any():
        dx = np.array([x[resolved].min(), x[resolved].max()])
        intercept = float(np.mean(y[resolved]) - expected_slope * np.mean(x[resolved]))
        ax.plot(
            dx,
            expected_slope * dx + intercept,
            color="mediumblue",
            label=f"{expected_slope:.1f}$x + c$",
        )
    if (~resolved).any():
        ax.scatter(
            x[~resolved], y[~resolved], marker="x", color="0.65", label="Roundoff"
        )
    ax.legend()
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(err_label)

    mode = "round-trip" if backward else "forward"
    fit_text = "unresolved" if measured is None else f"{measured:.6f}"
    print(
        f"{name} {mode} slope: {fit_text} (expected {expected_slope:g}; "
        f"{resolved.sum()}/{len(hs)} points above Decimal roundoff)",
        flush=True,
    )
    return {
        "expected_slope": expected_slope,
        "measured_slope": measured,
        "fit_points": int(resolved.sum()),
        "resolved": resolved.tolist(),
        "errors": [str(error) for error in errors],
        "log10_errors": y.tolist(),
        "roundoff_thresholds": [str(cutoff) for cutoff in roundoff],
    }


def plot_grid(min_power, max_power, precision, output_dir):
    with localcontext() as ctx:
        ctx.prec = precision
        recurrences = make_recurrences()
        counts = [2**power for power in range(min_power, max_power + 1)]
        hs = [Decimal(1) / count for count in counts]
        exact = Decimal(1) / Decimal(3).sqrt()
        fig, axes = plt.subplots(
            len(recurrences), 2, figsize=(10, 3.2 * len(recurrences))
        )
        results = {
            "arithmetic": "decimal.Decimal",
            "precision_digits": precision,
            "equation": "y' = -y^3",
            "t0": "0",
            "t1": "1",
            "y0": "1",
            "min_power": min_power,
            "max_power": max_power,
            "h": [str(h) for h in hs],
            "solvers": {},
        }

        for i, recurrence in enumerate(recurrences):
            values = [solve_roundtrip(recurrence, count) for count in counts]
            forward_errors = [abs(y - exact) for y, _ in values]
            backward_errors = [abs(y - 1) for _, y in values]
            method_results = {
                "coefficients": {
                    "A": [str(a) for a in recurrence.A],
                    "B": [str(b) for b in recurrence.B],
                },
            }
            for j, (mode, errors, rate) in enumerate(
                (
                    ("forward", forward_errors, 2),
                    ("roundtrip", backward_errors, recurrence.antisymmetric_order),
                )
            ):
                method_results[mode] = plot_curve(
                    recurrence.name,
                    hs,
                    errors,
                    rate,
                    precision,
                    axes[i, j],
                    backward=bool(j),
                )
                set_method_title(
                    axes[i, j], recurrence.antisymmetric_order, backward=bool(j)
                )
            results["solvers"][recurrence.name] = method_results

        fig.suptitle(rf"$\dot{{y}} = -y^3$, $y(0) = 1$, {precision}-digit arithmetic")
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "solver_convergence_decimal.png"
        fig.savefig(output_path, dpi=200)
        fig.savefig(output_path.with_suffix(".pdf"))
        plt.close(fig)
        output_path.with_suffix(".json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--min-power", type=int, default=2)
    parser.add_argument("--max-power", type=int, default=12)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    args = parser.parse_args()
    if args.precision < 16:
        parser.error("precision must be at least 16 decimal digits.")
    if args.min_power < 0 or args.max_power < args.min_power:
        parser.error("Require 0 <= min-power <= max-power.")
    return args


def main():
    args = parse_args()
    output_path = plot_grid(
        args.min_power, args.max_power, args.precision, args.output_dir
    )
    print(f"Saved {output_path} (also .pdf and .json)")


if __name__ == "__main__":
    main()
