## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## personal
from jormi.ww_io import manage_io
from jormi.ww_plots import add_color, annotate_axis, color_palettes, manage_plots
from jormi.ww_types import box_positions

## local
import plot_helpers

##
## === GLOBAL PARAMS
##

X_MIN, X_MAX = -1.5, 1.0
Y_MIN, Y_MAX = 2.95, 3.8

##
## === PIPELINE STAGES
##


def plot_suites(
    *,
    ax: Any,
    suite_stats_list: list[plot_helpers.SuiteStats],
    palette: color_palettes.SequentialPalette,
) -> None:
    for suite in suite_stats_list:
        print("Looking at:", suite.suite_name)
        marker, zorder = plot_helpers.get_suite_style(suite.suite_name)
        color = palette.mpl_cmap(
            palette.mpl_norm(
                suite.p_nl.p50,
            ),
        )
        plot_helpers.plot_suite_errorbar(
            ax=ax,
            x=suite.log10_Mach.p50,
            y=suite.log10_Re.p50,
            x_lo=suite.log10_Mach.std_lo,
            x_hi=suite.log10_Mach.std_hi,
            y_lo=suite.log10_Re.std_lo,
            y_hi=suite.log10_Re.std_hi,
            marker=marker,
            color=color,
            zorder=zorder,
        )


def style_axis(
    *,
    ax: Any,
    palette: color_palettes.SequentialPalette,
) -> None:
    ax.set_xlim((X_MIN, X_MAX))
    ax.set_ylim((Y_MIN, Y_MAX))
    ax.axvline(x=0, ls="--", color="black", lw=1.5, zorder=5)
    ax.set_xlabel(r"$\log_{10}(\mathcal{M})$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{Re})$")
    cbar = add_color.add_colorbar(
        ax=ax,
        palette=palette,
        label=r"$p_\mathrm{nl}$",
        cbar_side=box_positions.Positions.Side.Top,
        label_size=24,
    )
    cbar_ticks = [1.0, 1.25, 1.5, 1.75, 2.0]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{_tick}" for _tick in cbar_ticks])
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=plot_helpers.RESOLUTION_LEGEND_ARTISTS,
        labels=plot_helpers.RESOLUTION_LEGEND_LABELS,
        colors=["k"] * 3,
        marker_size=8,
        line_width=1.5,
        text_size=16,
        text_color="k",
        anchor_at_corner=box_positions.MPLPositions.Anchor.Corner.TopLeft,
        anchor_point=(0.0, 0.95),
    )


##
## === MAIN PROGRAM
##


def main() -> None:
    figures_dir, datasets_dir = plot_helpers.resolve_paper_dirs(
        Path(
            __file__,
        ),
    )
    manage_io.create_directory(figures_dir)
    suite_stats_list = plot_helpers.load_suite_stats(datasets_dir)
    palette = color_palettes.SequentialPalette.from_name(
        palette_name="purple-white-green",
        value_range=(1.0, 2.0),
    )
    fig, ax = manage_plots.create_figure()
    plot_suites(
        ax=ax,
        suite_stats_list=suite_stats_list,
        palette=palette,
    )
    style_axis(
        ax=ax,
        palette=palette,
    )
    manage_plots.save_figure(
        fig=fig,
        fig_path=figures_dir / "nl_exponent.pdf",
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
