## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy

## personal
from jormi.ww_io import manage_io
from jormi.ww_data import fit_series
from jormi.ww_plots import manage_plots, annotate_axis, add_color
from jormi.ww_data.series_types import GaussianSeries
from jormi.ww_plots.color_palettes import SequentialPalette

## local
import plot_helpers

##
## === GLOBAL PARAMS
##

X_MIN, X_MAX = -1.5, 1.0
AX0_Y_MIN, AX0_Y_MAX = -6.5, 0.0
AX1_Y_MIN, AX1_Y_MAX = 0.0, 2.0

##
## === LOCAL HELPERS
##


def generate_line(
    x_start: float,
    y_start: float,
    slope: float,
    line_length: float,
    domain_bounds: tuple[float, float, float, float],
    domain_aspect_ratio: float = 1.0,
    num_points: int = 2,
    direction: float = 1,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    x_min, x_max, y_min, y_max = domain_bounds
    data_aspect_ratio = (x_max - x_min) / (y_max - y_min)
    scale_x = 1.0
    scale_y = data_aspect_ratio / domain_aspect_ratio
    angle = numpy.arctan2(slope * scale_y, scale_x)
    x_step = line_length * numpy.cos(angle)
    y_step = line_length * numpy.sin(angle) / scale_y
    x_end = x_start + numpy.sign(direction) * x_step
    y_end = y_start + numpy.sign(direction) * y_step
    xs = numpy.linspace(x_start, x_end, num_points)
    ys = numpy.linspace(y_start, y_end, num_points)
    return xs, ys


##
## === PIPELINE STAGES
##


def plot_suites(
    *,
    axs: Any,
    suite_stats_list: list[plot_helpers.SuiteStats],
    palette: SequentialPalette,
) -> None:
    for suite in suite_stats_list:
        print("Looking at:", suite.suite_name)
        marker, zorder = plot_helpers.get_suite_style(suite.suite_name)
        marker_color = palette.mpl_cmap(palette.mpl_norm(suite.log10_Re.p50))
        plot_helpers.plot_suite_errorbar(
            ax=axs[0],
            x=suite.log10_Mach.p50,
            y=suite.log10_alpha_nl.p50,
            x_lo=suite.log10_Mach.std_lo,
            x_hi=suite.log10_Mach.std_hi,
            y_lo=suite.log10_alpha_nl.std_lo,
            y_hi=suite.log10_alpha_nl.std_hi,
            marker=marker,
            color=marker_color,
            zorder=zorder,
        )
        plot_helpers.plot_suite_errorbar(
            ax=axs[1],
            x=suite.log10_Mach.p50,
            y=suite.log10_nl_duration_normed_by_t0.p50,
            x_lo=suite.log10_Mach.std_lo,
            x_hi=suite.log10_Mach.std_hi,
            y_lo=suite.log10_nl_duration_normed_by_t0.std_lo,
            y_hi=suite.log10_nl_duration_normed_by_t0.std_hi,
            marker=marker,
            color=marker_color,
            zorder=zorder,
        )


def overlay_scalings(
    *,
    axs: Any,
    palette: SequentialPalette,
    suite_stats_list: list[plot_helpers.SuiteStats],
) -> None:
    x_values = numpy.linspace(-2, 2, 100)
    model_color = palette.mpl_cmap(0.7)
    ## subsonic growth rate
    subsonic_fit = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array([s.log10_Mach.p50 for s in suite_stats_list if s.log10_Mach.p50 < 0]),
            y_values=numpy.array([s.log10_alpha_nl.p50 for s in suite_stats_list if s.log10_Mach.p50 < 0]),
        ),
        fixed_slope=3,
    )
    subsonic_label = plot_helpers.format_fit_label(
        intercept_best=subsonic_fit.intercept.value,
        intercept_std=subsonic_fit.intercept.sigma,
        decimals=1,
    )
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=x_values,
        y_values=3 * x_values + subsonic_fit.intercept.value,
        linestyle="-",
        linewidth=1.5,
    )
    ## supersonic growth rate
    supersonic_fit = fit_series.fit_linear_model(
        GaussianSeries(
            x_values=numpy.array([s.log10_Mach.p50 for s in suite_stats_list if s.log10_Mach.p50 > -0.2]),
            y_values=numpy.array([s.log10_alpha_nl.p50 for s in suite_stats_list if s.log10_Mach.p50 > -0.2]),
        ),
    )
    supersonic_label = (
        rf"$10^{{{supersonic_fit.intercept.value:.1f} \pm {supersonic_fit.intercept.sigma:.1f}}}"
        rf"\,\mathcal{{M}}^{{{supersonic_fit.slope.value:.1f} \pm {supersonic_fit.slope.sigma:.1f}}}$"
    )
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=x_values,
        y_values=supersonic_fit.slope.value * x_values + supersonic_fit.intercept.value,
        linestyle="--",
        linewidth=1.5,
        zorder=3,
    )
    ## universal duration
    duration_fit = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array([s.log10_Mach.p50 for s in suite_stats_list]),
            y_values=numpy.array([s.log10_nl_duration_normed_by_t0.p50 for s in suite_stats_list]),
            y_sigmas=numpy.array([s.log10_nl_duration_normed_by_t0.std_hi for s in suite_stats_list]),
        ),
        fixed_slope=0,
    )
    duration_label = plot_helpers.format_fit_label(
        intercept_best=duration_fit.intercept.value,
        intercept_std=duration_fit.intercept.sigma,
        decimals=1,
    )
    axs[1].axhline(y=duration_fit.intercept.value, color="black", linestyle="-", linewidth=1.5, zorder=-1)
    annotate_axis.add_text(
        ax=axs[1],
        x_pos=0.035,
        y_pos=0.375,
        label=duration_label + r"$\, t_0 / t_\mathrm{sc}$",
        text_size=20,
        x_alignment="left",
        y_alignment="center",
    )
    ## reference models (xu & lazarian; beresnyak 2012)
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=x_values,
        y_values=numpy.log10(3 / 38 * 2) + 3 * x_values,
        color=model_color,  # type: ignore[arg-type]
        linestyle="-",
        linewidth=1.75,
        alpha=1.0,
        zorder=1,
    )
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=x_values,
        y_values=numpy.log10(0.05 * 2) + 3 * x_values,
        color=model_color,  # type: ignore[arg-type]
        linestyle="--",
        linewidth=1.75,
        alpha=1.0,
        zorder=1,
    )
    ## legends
    annotate_axis.add_custom_legend(
        ax=axs[0],
        artists=["--", "-"],
        colors=["black", "black"],
        labels=[
            supersonic_label,
            subsonic_label + r"$\, \mathcal{M}^3$",
        ],
        marker_size=8,
        line_width=1.5,
        text_size=16,
        text_color="k",
        anchor_at_corner="lower right",
        anchor_point=(1.0, 0.025),
        num_cols=1,
        spacing=0.625,
        frame_alpha=0.75,
    )
    annotate_axis.add_custom_legend(
        ax=axs[0],
        artists=["-", "--"],
        colors=[model_color, model_color],  # type: ignore[list-item]
        labels=[
            r"$(3/38) \, u_0^3 / \ell_0$",
            r"$0.05 \, u_0^3 / \ell_0$",
        ],
        marker_size=8,
        line_width=1.5,
        text_size=16,
        text_color="k",
        anchor_at_corner="upper left",
        anchor_point=(0.0, 0.99),
        num_cols=1,
        spacing=0.5,
    )


def style_axes(
    *,
    axs: Any,
    palette: SequentialPalette,
) -> None:
    axs[0].set_xticklabels([])
    axs[1].set_xlabel(r"$\log_{10}(\mathcal{M})$")
    axs[0].set_ylabel(r"$\log_{10}(\alpha_{\rm nl})$")
    axs[1].set_ylabel(r"$\log_{10}\big((t_{\rm sat} - t_{\rm nl}) / t_0\big)$")
    axs[0].set_xlim([X_MIN, X_MAX])
    axs[1].set_xlim([X_MIN, X_MAX])
    axs[0].set_ylim([AX0_Y_MIN, AX0_Y_MAX])
    axs[1].set_ylim([AX1_Y_MIN, AX1_Y_MAX])
    axs[0].axvline(x=0, color="black", linestyle=":", linewidth=1.5)
    axs[1].axvline(x=0, color="black", linestyle=":", linewidth=1.5)
    cbar = add_color.add_colorbar(
        ax=axs[0],
        palette=palette,
        label=r"$\log_{10}(\mathrm{Re})$",
        cbar_side="top",
        cbar_pad=0.015,
        label_size=24,
    )
    cbar_ticks = [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in cbar_ticks])
    annotate_axis.add_custom_legend(
        ax=axs[1],
        artists=plot_helpers.RESOLUTION_LEGEND_ARTISTS,
        labels=plot_helpers.RESOLUTION_LEGEND_LABELS,
        colors=["k"] * 3,
        marker_size=8,
        line_width=1.5,
        text_size=16,
        text_color="k",
        anchor_at_corner="upper left",
        anchor_point=(0.0, 1.0),
        num_cols=3,
        spacing=0.0,
    )


##
## === MAIN PROGRAM
##


def main() -> None:
    figures_dir, datasets_dir = plot_helpers.resolve_paper_dirs(Path(__file__))
    manage_io.init_directory(figures_dir)
    suite_stats_list = plot_helpers.load_suite_stats(datasets_dir)
    palette = SequentialPalette.from_name(
        palette_name="white-brown",
        value_range=(3.1, 3.7),
        palette_range=(0.0, 0.7),
    )
    fig, axs = manage_plots.create_figure(
        num_rows=2,
        num_cols=1,
    )
    axs = axs[:, 0]
    plot_suites(
        axs=axs,
        suite_stats_list=suite_stats_list,
        palette=palette,
    )
    overlay_scalings(
        axs=axs,
        palette=palette,
        suite_stats_list=suite_stats_list,
    )
    style_axes(
        axs=axs,
        palette=palette,
    )
    manage_plots.save_figure(
        fig=fig,
        fig_path=figures_dir / "nl_scalings.pdf",
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
