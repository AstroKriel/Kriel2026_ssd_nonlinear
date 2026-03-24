## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import manage_io, json_io
from jormi.ww_data import fit_series
from jormi.ww_plots import manage_plots, style_plots, annotate_axis, add_color
from jormi.ww_data.series_types import GaussianSeries
from jormi.ww_plots.color_palettes import SequentialPalette

##
## === GLOBAL PARAMS
##

X_MIN, X_MAX = -1.5, 1.0
AX0_Y_MIN, AX0_Y_MAX = -6.5, 0.0
AX1_Y_MIN, AX1_Y_MAX = 0.0, 2.0


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
    ## angle in screen space
    angle = numpy.arctan2(slope * scale_y, scale_x)
    x_step = line_length * numpy.cos(angle)
    y_step = line_length * numpy.sin(angle) / scale_y
    x_end = x_start + numpy.sign(direction) * x_step
    y_end = y_start + numpy.sign(direction) * y_step
    xs = numpy.linspace(x_start, x_end, num_points)
    ys = numpy.linspace(y_start, y_end, num_points)
    return xs, ys


##
## === MAIN PROGRAM
##


def main() -> None:
    ## define paths
    script_dir = Path(__file__).parent
    figures_dir = (script_dir / ".." / ".." / "figures").resolve()
    manage_io.init_directory(figures_dir)
    fig_path = figures_dir / "nl_scalings.pdf"
    dataset_dir = (script_dir / ".." / ".." / "datasets" / "summary.json").resolve()
    dataset = json_io.read_json_file_into_dict(dataset_dir)
    ## setup figure
    style_plots.set_theme()
    fig, axs = manage_plots.create_figure(num_rows=2, num_cols=1)
    axs = axs[:, 0]
    ## define custom colormap
    palette = SequentialPalette.from_name(
        palette_name="white-brown",
        value_range=(3.1, 3.7),
        palette_range=(0.0, 0.7),
    )
    model_color = palette.mpl_cmap(0.7)
    ## initialise data we want to fit to
    coords_to_fit = []
    ## loop over and plot each ensemble-averaged simulation suite
    for suite_name, suite_stats in dataset.items():
        print("Looking at:", suite_name)
        ## extract measured stats
        log10_Mach = suite_stats["measured"]["log10_Mach"]
        log10_Re = suite_stats["measured"]["log10_Re"]
        log10_alpha_nl = suite_stats["measured"]["log10_alpha_nl"]
        log10_nl_duration_normed_by_t0 = suite_stats["measured"]["log10_nl_duration_normed_by_t0"]
        ## define look of marker
        marker_color = palette.mpl_cmap(palette.mpl_norm(log10_Re["p50"]))
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
        axs[0].errorbar(
            log10_Mach["p50"],
            log10_alpha_nl["p50"],
            xerr=[
                [log10_Mach["std_lo"]],
                [log10_Mach["std_hi"]],
            ],
            yerr=[
                [log10_alpha_nl["std_lo"]],
                [log10_alpha_nl["std_hi"]],
            ],
            fmt=marker,
            markerfacecolor=marker_color,
            zorder=zorder,
            markeredgecolor="black",
            ecolor="black",
            markersize=10,
            linewidth=2,
            capsize=3,
        )
        axs[1].errorbar(
            log10_Mach["p50"],
            log10_nl_duration_normed_by_t0["p50"],
            xerr=[
                [log10_Mach["std_lo"]],
                [log10_Mach["std_hi"]],
            ],
            yerr=[
                [log10_nl_duration_normed_by_t0["std_lo"]],
                [log10_nl_duration_normed_by_t0["std_hi"]],
            ],
            fmt=marker,
            markerfacecolor=marker_color,
            zorder=zorder,
            markeredgecolor="black",
            ecolor="black",
            markersize=10,
            linewidth=2,
            capsize=3,
        )
        ## collect data we want to fit to
        coords_to_fit.append(
            (
                numpy.float64(log10_Mach["p50"]),
                numpy.float64(log10_alpha_nl["p50"]),
                numpy.float64(log10_nl_duration_normed_by_t0["p50"]),
                numpy.float64(log10_nl_duration_normed_by_t0["std_hi"]),  # assume summetric uncertainty
            ),
        )
    ## label
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
    ## fit and overlay scalings
    x_values = numpy.linspace(-2, 2, 100)
    ## subsonic growth rate
    subsonic_fit_results = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array([coord[0] for coord in coords_to_fit if coord[0] < 0]),
            y_values=numpy.array([coord[1] for coord in coords_to_fit if coord[0] < 0]),
        ),
        fixed_slope=3,
    )
    subsonic_a1_ave = subsonic_fit_results.intercept.value
    subsonic_a1_std = subsonic_fit_results.intercept.sigma
    subsonic_label = format_fit_label(
        intercept_best=subsonic_a1_ave,
        intercept_std=subsonic_a1_std,
        decimals=1,
    )
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=x_values,
        y_values=3 * x_values + subsonic_a1_ave,
        linestyle="-",
        linewidth=1.5,
    )
    ## supersonic growth rate
    supersonic_fit_results = fit_series.fit_linear_model(
        GaussianSeries(
            x_values=numpy.array([coord[0] for coord in coords_to_fit if -0.2 < coord[0]]),
            y_values=numpy.array([coord[1] for coord in coords_to_fit if -0.2 < coord[0]]),
        ),
    )
    supersonic_a0_ave = supersonic_fit_results.slope.value
    supersonic_a0_std = supersonic_fit_results.slope.sigma
    supersonic_a1_ave = supersonic_fit_results.intercept.value
    supersonic_a1_std = supersonic_fit_results.intercept.sigma
    supersonic_label = rf"$10^{{{supersonic_a1_ave:.1f} \pm {supersonic_a1_std:.1f}}}\,\mathcal{{M}}^{{{supersonic_a0_ave:.1f} \pm {supersonic_a0_std:.1f}}}$"
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=x_values,
        y_values=supersonic_a0_ave * x_values + supersonic_a1_ave,
        linestyle="--",
        linewidth=1.5,
        zorder=3,
    )
    ## universal duration
    duration_fit_results = fit_series.fit_line_with_fixed_slope(
        GaussianSeries(
            x_values=numpy.array([coord[0] for coord in coords_to_fit]),
            y_values=numpy.array([coord[2] for coord in coords_to_fit]),
            y_sigmas=numpy.array([coord[3] for coord in coords_to_fit]),
        ),
        fixed_slope=0,
    )
    duration_a1_ave = duration_fit_results.intercept.value
    duration_a1_std = duration_fit_results.intercept.sigma
    axs[1].axhline(y=duration_a1_ave, color="black", linestyle="-", linewidth=1.5, zorder=-1)
    duration_label = format_fit_label(
        intercept_best=duration_a1_ave,
        intercept_std=duration_a1_std,
        decimals=1,
    )
    annotate_axis.add_text(
        ax=axs[1],
        x_pos=0.035,
        y_pos=0.375,
        label=duration_label + r"$\, t_0 / t_\mathrm{sc}$",
        text_size=20,
        x_alignment="left",
        y_alignment="center",
    )
    ## annotate reference models
    ## xu and lazarian
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=x_values,
        y_values=numpy.log10(3 / 38 * 2) + 3 * x_values, # scaled by 1/ell_0 = 2.0
        color=model_color,  # type: ignore[arg-type]
        linestyle="-",
        linewidth=1.75,
        alpha=1.0,
        zorder=1,
    )
    ## beresnyak 2012
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=x_values,
        y_values=numpy.log10(0.05 * 2) + 3 * x_values, # scaled by 1/ell_0 = 2.0
        color=model_color,  # type: ignore[arg-type]
        linestyle="--",
        linewidth=1.75,
        alpha=1.0,
        zorder=1,
    )
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
        artists=[
            "-",
            "--",
        ],
        colors=[
            model_color,  # type: ignore[list-item]
            model_color,  # type: ignore[list-item]
        ],
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
    # ## schleicher
    # guide_x0 = 0.3
    # guide_y0 = -3.75
    # guide_length = 0.5
    # ax0_bounds = (X_MIN, X_MAX, AX0_Y_MIN, AX0_Y_MAX)
    # guide_x_y4, guide_y_y4 = generate_line(
    #     x_start=guide_x0,
    #     y_start=guide_y0,
    #     slope=4,
    #     line_length=guide_length,
    #     domain_bounds=ax0_bounds,
    #     domain_aspect_ratio=6 / 4,
    # )
    # annotate_axis.overlay_curve(
    #     ax=axs[0],
    #     x_values=guide_x_y4,
    #     y_values=guide_y_y4,
    #     color=guide_color,
    #     linestyle="--",
    #     linewidth=1.85,
    #     alpha=1.0,
    #     zorder=1,
    # )
    # annotate_axis.add_text(
    #     ax=axs[0],
    #     x_pos=0.85,
    #     y_pos=0.475,
    #     label=r"$\mathcal{M}^{4}$",
    #     x_alignment="center",
    #     y_alignment="center",
    #     font_color=guide_color,
    #     fontsize=20,
    # )
    ## add colorbar
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
    cbar.set_ticklabels([f"{cbar_tick:.1f}" for cbar_tick in cbar_ticks])
    annotate_axis.add_custom_legend(
        ax=axs[1],
        artists=["o", "s", "D"],
        labels=[r"$288^3$", r"$576^3$", r"$1152^3$"],
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
    manage_plots.save_figure(fig, fig_path)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
