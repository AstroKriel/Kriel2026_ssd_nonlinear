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
from jormi.ww_io import manage_io, json_io
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


def load_dataset(
    datasets_dir: Path,
) -> dict:
    dataset_path = datasets_dir / "summary.json"
    return json_io.read_json_file_into_dict(dataset_path)


def plot_suites(
    *,
    ax: Any,
    dataset: dict,
    palette: DivergingPalette,
) -> list[tuple]:
    coords_to_fit = []
    for suite_name, suite_stats in dataset.items():
        print("Looking at:", suite_name)
        log10_Mach = suite_stats["measured"]["log10_Mach"]
        log10_Re = suite_stats["measured"]["log10_Re"]
        log10_gamma_exp_times_t0 = suite_stats["measured"]["log10_gamma_exp_times_t0"]
        marker, zorder = plot_helpers.get_suite_style(suite_name)
        color = palette.mpl_cmap(palette.mpl_norm(log10_Mach["p50"]))
        plot_helpers.plot_suite_errorbar(
            ax=ax,
            x=log10_Re["p50"],
            y=log10_gamma_exp_times_t0["p50"],
            x_lo=log10_Re["std_lo"],
            x_hi=log10_Re["std_hi"],
            y_lo=log10_gamma_exp_times_t0["std_lo"],
            y_hi=log10_gamma_exp_times_t0["std_hi"],
            marker=marker,
            color=color,
            zorder=zorder,
        )
        coords_to_fit.append(
            (
                numpy.float64(log10_Mach["p50"]),
                numpy.float64(log10_Re["p50"]),
                numpy.float64(log10_gamma_exp_times_t0["p50"]),
            ),
        )
    return coords_to_fit


def overlay_scalings(
    *,
    ax: Any,
    palette: DivergingPalette,
    coords_to_fit: list[tuple],
) -> None:
    x_values = numpy.linspace(3, 4, 100)
    rotation_bounds = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    ## subsonic scaling
    subsonic_fit = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array([c[1] for c in coords_to_fit if c[0] < 0]),
            y_values=numpy.array([c[2] for c in coords_to_fit if c[0] < 0]),
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
        text_color=palette.mpl_cmap(palette.mpl_norm(-1.0)),  # type: ignore[arg-type]
    )
    ## supersonic scaling
    supersonic_fit = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array([c[1] for c in coords_to_fit if 0 < c[0]]),
            y_values=numpy.array([c[2] for c in coords_to_fit if 0 < c[0]]),
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
        text_color=palette.mpl_cmap(palette.mpl_norm(1.0)),  # type: ignore[arg-type]
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
    ax.set_xticklabels(f"{x:.1f}" for x in x_ticks)
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
    dataset = load_dataset(datasets_dir)
    palette = DivergingPalette.from_name(
        palette_name="blue-white-red",
        value_range=(-1.0, 1.0),
        mid_value=0.0,
    )
    fig, ax = manage_plots.create_figure()
    coords_to_fit = plot_suites(ax=ax, dataset=dataset, palette=palette)
    overlay_scalings(ax=ax, palette=palette, coords_to_fit=coords_to_fit)
    style_axis(ax=ax, palette=palette)
    manage_plots.save_figure(fig, figures_dir / "gamma_exp_scaling.pdf")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
