## { SCRIPT

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import manage_io, json_io
from jormi.ww_data import fit_series
from jormi.ww_plots import manage_plots, style_plots, annotate_axis, add_color
from jormi.ww_data.series_types import GaussianSeries
from jormi.ww_plots.color_palettes import DivergingPalette

x_min, x_max = 3.05, 3.75
y_min, y_max = -0.4, 0.45


def format_fit_label(
    intercept_best: float,
    intercept_std: float,
    decimals: int = 2,
) -> str:
    coefficient = 10**intercept_best
    coefficient_std = numpy.log(10) * coefficient * intercept_std
    exponent = int(numpy.floor(numpy.log10(coefficient)))
    significand = coefficient / (10**exponent)
    significand_std = coefficient_std / (10**exponent)
    if exponent == 0:
        label = rf"$({significand:.{decimals}f} \pm {significand_std:.{decimals}f})$"
    else:
        label = rf"$({significand:.{decimals}f} \pm {significand_std:.{decimals}f}) \!\times 10^{{{exponent}}}$"
    return label


def main() -> None:
    ## define paths
    script_dir = Path(__file__).parent
    figures_dir = (script_dir / ".." / ".." / "figures").resolve()
    manage_io.init_directory(figures_dir)
    fig_path = figures_dir / "gamma_exp_scaling.pdf"
    dataset_dir = (script_dir / ".." / ".." / "datasets" / "summary.json").resolve()
    dataset = json_io.read_json_file_into_dict(dataset_dir)
    ## setup figure
    style_plots.set_theme()
    fig, ax = manage_plots.create_figure()
    ## define custom colormap
    palette = DivergingPalette.from_name(
        palette_name="blue-white-red",
        value_range=(-1.0, 1.0),
        mid_value=0.0,
    )
    ## initialise data we want to fit to
    coords_to_fit = []
    ## loop over and plot each ensemble-averaged simulation suite
    for suite_name, suite_stats in dataset.items():
        print("Looking at:", suite_name)
        ## extract measured stats
        log10_Mach = suite_stats["measured"]["log10_Mach"]
        log10_Re = suite_stats["measured"]["log10_Re"]
        log10_gamma_exp_times_t0 = suite_stats["measured"]["log10_gamma_exp_times_t0"]
        ## tweak plot params
        color = palette.mpl_cmap(palette.mpl_norm(log10_Mach["p50"]))
        if "288" in suite_name:
            marker = "o"
            zorder = 1
        elif "576" in suite_name:
            marker = "s"
            zorder = 3
        elif "1152" in suite_name:
            marker = "D"
            zorder = 5
        else:
            print("Could not determine resolution for:", suite_name)
            continue
        ## plot
        ax.errorbar(
            log10_Re["p50"],
            log10_gamma_exp_times_t0["p50"],
            xerr=[
                [log10_Re["std_lo"]],
                [log10_Re["std_hi"]],
            ],
            yerr=[
                [log10_gamma_exp_times_t0["std_lo"]],
                [log10_gamma_exp_times_t0["std_hi"]],
            ],
            fmt=marker,
            markerfacecolor=color,
            zorder=zorder,
            markeredgecolor="black",
            ecolor="black",
            markersize=10,
            lw=2,
            capsize=3,
        )
        ## collect data we want to fit to
        coords_to_fit.append(
            (
                numpy.float64(log10_Mach["p50"]),
                numpy.float64(log10_Re["p50"]),
                numpy.float64(log10_gamma_exp_times_t0["p50"]),
            ),
        )
    ## label
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    x_ticks = [3.1, 3.3, 3.5, 3.7]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(f"{x_tick:.1f}" for x_tick in x_ticks)
    ax.set_xlabel(r"$\log_{10}(\mathrm{Re})$")
    ax.set_ylabel(r"$\log_{10}(\gamma_\mathrm{exp} \,t_0)$")
    ## fit and overlay scalings
    x_values = numpy.linspace(3, 4, 100)
    rotation_bounds = (x_min, x_max, y_min, y_max)
    ## subsonic scaling
    subsonic_fit_results = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array([coord[1] for coord in coords_to_fit if coord[0] < 0]),
            y_values=numpy.array([coord[2] for coord in coords_to_fit if coord[0] < 0]),
        ),
        fixed_slope=0.5,
    )
    subsonic_a1_ave = subsonic_fit_results.intercept.value
    subsonic_a1_std = subsonic_fit_results.intercept.sigma
    annotate_axis.overlay_curve(
        ax=ax,
        x_values=x_values,
        y_values=0.5 * x_values + subsonic_a1_ave,
        linestyle="-",
        linewidth=1.5,
    )
    subsonic_label = format_fit_label(
        intercept_best=subsonic_a1_ave,
        intercept_std=subsonic_a1_std,
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
    supersonic_fit_results = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array([coord[1] for coord in coords_to_fit if 0 < coord[0]]),
            y_values=numpy.array([coord[2] for coord in coords_to_fit if 0 < coord[0]]),
        ),
        fixed_slope=1 / 3,
    )
    supersonic_a1_ave = supersonic_fit_results.intercept.value
    supersonic_a1_std = supersonic_fit_results.intercept.sigma
    annotate_axis.overlay_curve(
        ax=ax,
        x_values=x_values,
        y_values=(1 / 3) * x_values + supersonic_a1_ave,
        linestyle="--",
        linewidth=1.5,
    )
    supersonic_label = format_fit_label(
        intercept_best=supersonic_a1_ave,
        intercept_std=supersonic_a1_std,
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
    ## add other labels
    add_color.add_colorbar(
        ax=ax,
        palette=palette,
        label=r"$\log_{10}(\mathcal{M})$",
        cbar_side="top",
        label_size=24,
    )
    annotate_axis.add_custom_legend(
        ax=ax,
        artists=[
            "o",
            "D",
            "s",
        ],
        labels=[
            r"$288^3$",
            r"$1152^3$",
            r"$576^3$",
        ],
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
    manage_plots.save_figure(fig, fig_path)


if __name__ == "__main__":
    main()

## } SCRIPT
