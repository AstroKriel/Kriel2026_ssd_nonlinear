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
from jormi.ww_plots.color_palettes import DivergingPalette

## local
import plot_helpers

##
## === GLOBAL PARAMS
##

X_MIN, X_MAX = 3.05, 3.75
Y_MIN, Y_MAX = -0.4, 0.45

##
## === PIPELINE STAGES
##


def plot_suites(
    *,
    ax: Any,
    suite_stats_list: list[plot_helpers.SuiteStats],
    palette: DivergingPalette,
) -> None:
    for suite in suite_stats_list:
        print("Looking at:", suite.suite_name)
        marker, zorder = plot_helpers.get_suite_style(suite.suite_name)
        color = palette.mpl_cmap(palette.mpl_norm(suite.log10_Mach.p50))
        plot_helpers.plot_suite_errorbar(
            ax=ax,
            x=suite.log10_Re.p50,
            y=suite.log10_gamma_exp_times_t0.p50,
            x_lo=suite.log10_Re.std_lo,
            x_hi=suite.log10_Re.std_hi,
            y_lo=suite.log10_gamma_exp_times_t0.std_lo,
            y_hi=suite.log10_gamma_exp_times_t0.std_hi,
            marker=marker,
            color=color,
            zorder=zorder,
        )


def overlay_scalings(
    *,
    ax: Any,
    palette: DivergingPalette,
    suite_stats_list: list[plot_helpers.SuiteStats],
) -> None:
    x_values = numpy.linspace(3, 4, 100)
    rotation_bounds = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    ## subsonic scaling
    subsonic_fit = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array(
                [
                    _suite_stats.log10_Re.p50
                    for _suite_stats in suite_stats_list
                    if _suite_stats.log10_Mach.p50 < 0
                ]
            ),
            y_values=numpy.array(
                [
                    _suite_stats.log10_gamma_exp_times_t0.p50
                    for _suite_stats in suite_stats_list
                    if _suite_stats.log10_Mach.p50 < 0
                ],
            ),
        ),
        fixed_slope=0.5,
    )
    annotate_axis.overlay_curve(
        ax=ax,
        x_values=x_values,
        y_values=0.5 * x_values + subsonic_fit.intercept.value,
        linestyle="-",
        linewidth=1.5,
    )
    subsonic_label = plot_helpers.format_fit_label(
        intercept_best=subsonic_fit.intercept.value,
        intercept_std=subsonic_fit.intercept.sigma,
        decimals=2,
    )
    subsonic_rotation = fit_series.get_line_angle(
        slope=0.5,
        domain_bounds=rotation_bounds,
        aspect_ratio=6 / 4,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.585,
        y_pos=0.665,
        label=subsonic_label + r"$\, \mathrm{Re}^{1/2}$",
        x_alignment="center",
        y_alignment="center",
        rotate_deg=subsonic_rotation,
        text_color=palette.mpl_cmap(palette.mpl_norm(-1.0)),
    )
    ## supersonic scaling
    supersonic_fit = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array(
                [
                    _suite_stats.log10_Re.p50
                    for _suite_stats in suite_stats_list
                    if _suite_stats.log10_Mach.p50 > 0
                ]
            ),
            y_values=numpy.array(
                [
                    _suite_stats.log10_gamma_exp_times_t0.p50
                    for _suite_stats in suite_stats_list
                    if _suite_stats.log10_Mach.p50 > 0
                ],
            ),
        ),
        fixed_slope=1 / 3,
    )
    annotate_axis.overlay_curve(
        ax=ax,
        x_values=x_values,
        y_values=(1 / 3) * x_values + supersonic_fit.intercept.value,
        linestyle="--",
        linewidth=1.5,
    )
    supersonic_label = plot_helpers.format_fit_label(
        intercept_best=supersonic_fit.intercept.value,
        intercept_std=supersonic_fit.intercept.sigma,
        decimals=2,
    )
    supersonic_rotation = fit_series.get_line_angle(
        slope=1 / 3.5,
        domain_bounds=rotation_bounds,
        aspect_ratio=6 / 4,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.665,
        y_pos=0.15,
        label=supersonic_label + r"$\, \mathrm{Re}^{1/3}$",
        x_alignment="center",
        y_alignment="center",
        rotate_deg=supersonic_rotation,
        text_color=palette.mpl_cmap(palette.mpl_norm(1.0)),
    )


def style_axis(
    *,
    ax: Any,
    palette: DivergingPalette,
) -> None:
    ax.set_xlim((X_MIN, X_MAX))
    ax.set_ylim((Y_MIN, Y_MAX))
    x_ticks = [3.1, 3.3, 3.5, 3.7]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(f"{_tick:.1f}" for _tick in x_ticks)
    ax.set_xlabel(r"$\log_{10}(\mathrm{Re})$")
    ax.set_ylabel(r"$\log_{10}(\gamma_\mathrm{exp} \,t_0)$")
    add_color.add_colorbar(
        ax=ax,
        palette=palette,
        label=r"$\log_{10}(\mathcal{M})$",
        cbar_side="top",
        label_size=24,
    )
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=plot_helpers.RESOLUTION_LEGEND_ARTISTS,
        labels=plot_helpers.RESOLUTION_LEGEND_LABELS,
        colors=["k"] * 3,
        marker_size=8,
        line_width=1.5,
        text_size=16,
        text_color="k",
        anchor_at_corner="upper left",
        anchor_point=(0.0, 1.0),
        num_cols=2,
        spacing=0.0,
    )


##
## === MAIN PROGRAM
##


def main() -> None:
    figures_dir, datasets_dir = plot_helpers.resolve_paper_dirs(Path(__file__))
    manage_io.init_directory(figures_dir)
    suite_stats_list = plot_helpers.load_suite_stats(datasets_dir)
    palette = DivergingPalette.from_name(
        palette_name="blue-white-red",
        value_range=(-1.0, 1.0),
        mid_value=0.0,
    )
    fig, ax = manage_plots.create_figure()
    plot_suites(
        ax=ax,
        suite_stats_list=suite_stats_list,
        palette=palette,
    )
    overlay_scalings(
        ax=ax,
        palette=palette,
        suite_stats_list=suite_stats_list,
    )
    style_axis(
        ax=ax,
        palette=palette,
    )
    manage_plots.save_figure(
        fig=fig,
        fig_path=figures_dir / "gamma_exp_scaling.pdf",
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
