"""Diagnostic views over the whole case ensemble.

Both views take one scalar per case - a roughness score, a predictor error, anything - and
place it on the RH18 scaling coordinates. The scatter is lossless and stays readable at a
few thousand cases; the binned view is easier to read but only alongside its count and
spread panels, which say whether a cell mean is standing on anything.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps
from matplotlib.colors import Normalize
from scipy.stats import binned_statistic_2d

from utility import calculate_B, calculate_f, compute_u_star, mean_over_last_days


GRID = "#e8e8e8"
AXIS = "#bdbdbd"
INK = "#4a4a4a"
EMPTY = "#f2f2f2"

X_LABEL = r"$\overline{H}\,f\,/\,u_*$"
Y_LABEL = r"$B$   [m$^2$ s$^{-3}$]"


def scaling_coordinates(
        case_dict: dict,
        bl_depth,
        dt: float,
        n_days: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """RH18 scaling coordinates per case: mean H f / u_star, and surface buoyancy flux.

    `bl_depth` is a (case, time) DataArray, so select a single bl_method before passing it.
    Mean H is taken directly over the last `n_days` rather than from a centred rolling mean,
    whose reflected padding would put a latitude dependent artifact straight into the x axis.
    """
    cases = [str(case) for case in bl_depth.coords['case'].values]

    x, y = np.empty(len(cases)), np.empty(len(cases))

    for i, case in enumerate(cases):
        specs = case_dict[case]
        mean_h = mean_over_last_days(
            bl_depth.sel(case=case).values, dt=dt, n_days=n_days
        )
        x[i] = mean_h * calculate_f(latitude=specs['lat']) / compute_u_star(tau=specs['tx'])
        y[i] = calculate_B(heat_fluxes=specs['heat_flux'])

    return x, y


def value_scale(
        values: np.ndarray,
        diverging: bool = False,
        cmap: str = None
) -> tuple[str, Normalize]:
    """One hue light to dark for magnitudes, warm against cool about zero for polarity.

    The diverging range is forced symmetric so the neutral midpoint lands exactly on zero;
    an asymmetric range would put the "no bias" colour somewhere that is not no bias.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        return (cmap or "Blues"), Normalize(vmin=0, vmax=1)

    if diverging:
        limit = np.max(np.abs(finite)) or 1.0
        return (cmap or "coolwarm"), Normalize(vmin=-limit, vmax=limit)

    return (cmap or "Blues"), Normalize(vmin=finite.min(), vmax=finite.max())


def discrete_bin_edges(values: np.ndarray) -> np.ndarray:
    """Edges midway between the distinct values, for an axis that is already categorical."""
    unique = np.unique(np.asarray(values)[np.isfinite(values)])

    if unique.size == 1:
        return np.array([unique[0] - 0.5, unique[0] + 0.5])

    midpoints = (unique[:-1] + unique[1:]) / 2

    return np.concatenate([
        [2 * unique[0] - midpoints[0]], midpoints, [2 * unique[-1] - midpoints[-1]]
    ])


def style_axes(ax, grid: bool = True) -> None:
    ax.spines[['top', 'right']].set_visible(False)

    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(AXIS)

    ax.tick_params(length=3, width=0.6, colors=INK, labelsize=9)

    if grid:
        ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)


def add_colorbar(mesh, ax, label: str):
    bar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label(label, color=INK, fontsize=9)
    bar.outline.set_linewidth(0)
    bar.ax.tick_params(length=3, width=0.6, colors=INK, labelsize=9)
    return bar


def scatter_cases(
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        label: str = "",
        diverging: bool = False,
        cmap: str = None,
        log_x: bool = True,
        jitter: float = 0.25,
        ax=None
):
    """One mark per case, coloured by `values`. Nothing is averaged away.

    `y` is categorical - one row per heat flux - so marks pile up along each row; `jitter`
    spreads them across a fraction of the row spacing to make local density legible. Look
    at this before reaching for the binned view: at a few thousand cases it is still
    readable, and it cannot hide structure the way a cell mean can.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    name, norm = value_scale(values=values, diverging=diverging, cmap=cmap)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

    if jitter > 0:
        unique = np.unique(y[np.isfinite(y)])
        spacing = np.min(np.diff(unique)) if unique.size > 1 else 1.0
        offsets = np.random.default_rng(0).uniform(-0.5, 0.5, size=y.size)
        y = y + offsets * jitter * spacing

    marks = ax.scatter(
        x, y,
        c=values, cmap=name, norm=norm,
        s=18, alpha=0.85, linewidths=0.3, edgecolors="white"
    )

    if log_x:
        ax.set_xscale("log")

    style_axes(ax)
    ax.set_xlabel(X_LABEL, color=INK)
    ax.set_ylabel(Y_LABEL, color=INK)
    add_colorbar(mesh=marks, ax=ax, label=label)

    return ax


def binned_cases(
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        label: str = "",
        n_x_bins: int = 20,
        y_edges: np.ndarray = None,
        statistic: str = "mean",
        min_count: int = 3,
        diverging: bool = False,
        cmap: str = None,
        log_x: bool = True
):
    """Aggregate cases into cells, with the count and within-cell spread beside the statistic.

    The spread panel is the one that earns its place: mean(H) f / u_star collapses temp_grad,
    tx and lat onto a single axis, so every cell merges physically different setups. Tight
    spread means those two parameters really do control the response and the collapse is
    fair; wide spread means something outside them is driving it and the cell mean is hiding
    that. Cells holding fewer than `min_count` cases are left blank rather than drawn from
    too little.

    Returns the figure and the binned arrays, so the numbers stay readable without the
    colour scale.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[keep], y[keep], values[keep]

    if y_edges is None:
        y_edges = discrete_bin_edges(values=y)

    x_edges = (
        np.geomspace(x.min(), x.max(), n_x_bins + 1) if log_x
        else np.linspace(x.min(), x.max(), n_x_bins + 1)
    )
    bins = [x_edges, y_edges]

    stat = binned_statistic_2d(x, y, values, statistic=statistic, bins=bins).statistic
    count = binned_statistic_2d(x, y, values, statistic="count", bins=bins).statistic
    spread = binned_statistic_2d(x, y, values, statistic="std", bins=bins).statistic

    thin = count < min_count
    stat = np.where(thin, np.nan, stat)
    spread = np.where(thin, np.nan, spread)
    count = np.where(count == 0, np.nan, count)

    panels = [
        (stat, f"{statistic} {label}".strip(), diverging, cmap),
        (count, "cases per cell", False, "Greys"),
        (spread, f"spread within cell", False, None)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    for ax, (data, title, is_diverging, panel_cmap) in zip(axes, panels):
        name, norm = value_scale(values=data, diverging=is_diverging, cmap=panel_cmap)
        shading = colormaps[name].copy()
        shading.set_bad(EMPTY)

        mesh = ax.pcolormesh(x_edges, y_edges, data.T, cmap=shading, norm=norm)

        if log_x:
            ax.set_xscale("log")

        style_axes(ax, grid=False)
        ax.set_xlabel(X_LABEL, color=INK)
        ax.set_title(title, color=INK, fontsize=10, pad=8)
        add_colorbar(mesh=mesh, ax=ax, label="")

    axes[0].set_ylabel(Y_LABEL, color=INK)

    return fig, {
        "statistic": stat,
        "count": count,
        "spread": spread,
        "x_edges": x_edges,
        "y_edges": y_edges
    }
