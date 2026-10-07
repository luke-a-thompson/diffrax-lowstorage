"""The convergence chart style used by the Georax examples."""

import matplotlib.markers as mkr
import numpy as np

XLABEL = r"$\log_{10}(h)$"
YLABEL_FORWARD = r"$\log_{10}(\mathcal{E}(h))$"
YLABEL_BACKWARD = r"$\log_{10}(\overleftarrow{\mathcal{E}}(h))$"


def set_method_title(ax, antisymmetric_order, *, backward=False):
    """Use the mathematical panel titles from Georax's stochastic plots."""
    method = rf"\mathrm{{EES}}_\mathcal{{R}}(2,{antisymmetric_order})"
    error = r"\overleftarrow{\mathcal{E}}(h)" if backward else r"\mathcal{E}(h)"
    ax.set_title(rf"${error}$ for ${method}$")


def plot_curve(name, h, y, slope, ax, *, backward=False):
    """Plot log-errors and an expected-rate guide; fit above roundoff only."""
    x = np.log10(h)
    # Allow for roundoff accumulated across 1/h steps. A zero round-trip defect
    # means cancellation at machine precision, rather than an exact inverse.
    roundoff = 100 * np.finfo(np.float64).eps / np.sqrt(h)
    resolved = np.isfinite(y) & (y > np.log10(roundoff))
    measured = (
        float(np.polyfit(x[resolved], y[resolved], 1)[0])
        if resolved.sum() >= 2
        else None
    )
    err_label = YLABEL_BACKWARD if backward else YLABEL_FORWARD
    mode = "round-trip" if backward else "forward"
    fit_text = "unresolved" if measured is None else f"{measured:.6f}"
    print(
        f"{name} {mode} slope: {fit_text} (expected {slope:g}; "
        f"{resolved.sum()}/{len(h)} points above roundoff)",
        flush=True,
    )

    ax.scatter(
        x[resolved],
        y[resolved],
        marker=mkr.MarkerStyle("x", fillstyle="none"),
        color="crimson",
        label=err_label,
    )
    if resolved.any():
        dx = np.array([x[resolved].min(), x[resolved].max()])
        intercept = float(np.mean(y[resolved]) - slope * np.mean(x[resolved]))
        ax.plot(
            dx, slope * dx + intercept, color="mediumblue", label=f"{slope:.1f}$x + c$"
        )
    if (~resolved).any():
        ax.scatter(
            x[~resolved], y[~resolved], marker="x", color="0.65", label="Roundoff"
        )
    ax.legend()
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(err_label)

    return {
        "expected_slope": float(slope),
        "measured_slope": measured,
        "fit_points": int(resolved.sum()),
        "log10_errors": np.asarray(y).tolist(),
        "resolved": resolved.tolist(),
    }
