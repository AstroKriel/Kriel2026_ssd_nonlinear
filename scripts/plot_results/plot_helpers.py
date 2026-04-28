## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

## third-party
import numpy

## personal
from jormi.ww_io import json_io

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class MeasuredStat:
    p50: float
    std_lo: float
    std_hi: float


@dataclass(frozen=True)
class SuiteStats:
    suite_name: str
    log10_Mach: MeasuredStat
    log10_Re: MeasuredStat
    log10_gamma_exp_times_t0: MeasuredStat
    log10_alpha_nl: MeasuredStat
    log10_nl_duration_normed_by_t0: MeasuredStat
    p_nl: MeasuredStat


##
## === DATASET LOADER
##


def load_suite_stats(
    datasets_dir: Path,
) -> list[SuiteStats]:
    raw = json_io.read_json_file_into_dict(datasets_dir / "summary.json")
    result = []
    for suite_name, data in raw.items():
        m = data["measured"]
        result.append(
            SuiteStats(
                suite_name=suite_name,
                log10_Mach=MeasuredStat(
                    p50=m["log10_Mach"]["p50"],
                    std_lo=m["log10_Mach"]["std_lo"],
                    std_hi=m["log10_Mach"]["std_hi"],
                ),
                log10_Re=MeasuredStat(
                    p50=m["log10_Re"]["p50"],
                    std_lo=m["log10_Re"]["std_lo"],
                    std_hi=m["log10_Re"]["std_hi"],
                ),
                log10_gamma_exp_times_t0=MeasuredStat(
                    p50=m["log10_gamma_exp_times_t0"]["p50"],
                    std_lo=m["log10_gamma_exp_times_t0"]["std_lo"],
                    std_hi=m["log10_gamma_exp_times_t0"]["std_hi"],
                ),
                log10_alpha_nl=MeasuredStat(
                    p50=m["log10_alpha_nl"]["p50"],
                    std_lo=m["log10_alpha_nl"]["std_lo"],
                    std_hi=m["log10_alpha_nl"]["std_hi"],
                ),
                log10_nl_duration_normed_by_t0=MeasuredStat(
                    p50=m["log10_nl_duration_normed_by_t0"]["p50"],
                    std_lo=m["log10_nl_duration_normed_by_t0"]["std_lo"],
                    std_hi=m["log10_nl_duration_normed_by_t0"]["std_hi"],
                ),
                p_nl=MeasuredStat(
                    p50=m["p_nl"]["p50"],
                    std_lo=m["p_nl"]["std_lo"],
                    std_hi=m["p_nl"]["std_hi"],
                ),
            ),
        )
    return result


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
    exponent = int(
        numpy.floor(
            numpy.log10(
                coefficient,
            ),
        ),
    )
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
