"""The convergence chart style used by the Georax examples."""

from matplotlib.axes import Axes

XLABEL = r"$\log_{10}(h)$"
YLABEL_FORWARD = r"$\log_{10}(\mathcal{E}(h))$"
YLABEL_BACKWARD = r"$\log_{10}(\overleftarrow{\mathcal{E}}(h))$"


def set_method_title(
    ax: Axes, antisymmetric_order: int, *, backward: bool = False
) -> None:
    """Use the mathematical panel titles from Georax's convergence plots."""
    method = rf"\mathrm{{EES}}_\mathcal{{R}}(2,{antisymmetric_order})"
    error = r"\overleftarrow{\mathcal{E}}(h)" if backward else r"\mathcal{E}(h)"
    ax.set_title(rf"${error}$ for ${method}$")
