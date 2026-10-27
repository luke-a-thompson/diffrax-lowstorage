"""The convergence chart style used by the Georax examples."""

XLABEL = r"$\log_{10}(h)$"
YLABEL_FORWARD = r"$\log_{10}(\mathcal{E}(h))$"
YLABEL_BACKWARD = r"$\log_{10}(\overleftarrow{\mathcal{E}}(h))$"


def set_method_title(ax, antisymmetric_order, *, backward=False):
    """Use the mathematical panel titles from Georax's convergence plots."""
    method = rf"\mathrm{{EES}}_\mathcal{{R}}(2,{antisymmetric_order})"
    error = r"\overleftarrow{\mathcal{E}}(h)" if backward else r"\mathcal{E}(h)"
    ax.set_title(rf"${error}$ for ${method}$")
