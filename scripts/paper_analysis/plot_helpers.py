## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy

##
## === RESOLUTION STYLES
##

_RESOLUTION_STYLES: dict[str, tuple[str, int]] = {
    "288": ("o", 1),
    "576": ("s", 3),
    "1152": ("D", 5),
}

RESOLUTION_LEGEND_ARTISTS: list[str] = ["o", "s", "D"]
RESOLUTION_LEGEND_LABELS: list[str] = [r"$288^3$", r"$576^3$", r"$1152^3$"]


def get_suite_style(
    suite_name: str,
) -> tuple[str, int]:
    for res_str, style in _RESOLUTION_STYLES.items():
        if res_str in suite_name:
            return style
    raise ValueError(f"Could not determine resolution for: {suite_name}")


##
## === PATH HELPERS
##


def resolve_paper_dirs(
    script_path: Path,
) -> tuple[Path, Path]:
    """Return (figures_dir, datasets_dir) relative to the repo root."""
    root = (script_path.parent / ".." / "..").resolve()
    return root / "figures", root / "datasets"


##
## === PLOT HELPERS
##


def plot_suite_errorbar(
    *,
    ax: Any,
    x: float,
    y: float,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    marker: str,
    color: Any,
    zorder: int,
) -> None:
    ax.errorbar(
        x,
        y,
        xerr=[[x_lo], [x_hi]],
        yerr=[[y_lo], [y_hi]],
        fmt=marker,
        markerfacecolor=color,
        markeredgecolor="black",
        ecolor="black",
        markersize=10,
        lw=2,
        capsize=3,
        zorder=zorder,
    )


##
## === LABEL HELPERS
##


def format_fit_label(
    intercept_best: float,
    intercept_std: float | None,
    decimals: int = 2,
) -> str:
    coefficient = 10**intercept_best
    coefficient_std = numpy.log(10) * coefficient * (intercept_std if intercept_std is not None else 0.0)
    exponent = int(numpy.floor(numpy.log10(coefficient)))
    significand = coefficient / (10**exponent)
    significand_std = coefficient_std / (10**exponent)
    if exponent == 0:
        label = rf"$({significand:.{decimals}f} \pm {significand_std:.{decimals}f})$"
    elif exponent == 1:
        label = rf"$({significand:.{decimals}f} \pm {significand_std:.{decimals}f}) \!\times 10$"
    else:
        label = rf"$({significand:.{decimals}f} \pm {significand_std:.{decimals}f}) \!\times 10^{{{exponent}}}$"
    return label


## } MODULE
