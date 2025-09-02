## { SCRIPT

import numpy
import random
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from jormi.ww_io import json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_styler, annotate_axis, add_color

MODEL_TYPE = "linear"
# MODEL_TYPE = "quadratic"
# MODEL_TYPE = "free"

# BINNING_TYPE = "100bins"
BINNING_TYPE = "bin_per_t0"

x_min, x_max = -1.5, 1.0
y0_min, y0_max = -6.5, 0.0
y1_min, y1_max = 0.0, 2.0


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
    domain_bounds: float,
    domain_aspect_ratio: float = 1.0,
    num_points: float = 2,
    direction: float = 1,
):
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


def main():
    summary_path = Path(
        "/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_ssd_nl/datasets_v2/summary_stats.json",
    )
    all_results = json_files.read_json_file_into_dict(summary_path)

    plot_styler.apply_theme_globally()
    fig, axs = plot_manager.create_figure(num_rows=2)

    custom_cmap = LinearSegmentedColormap.from_list(
        name="white-brown",
        colors=["#fdfdfd", "#f49325", "#010101"],
        N=256,
    )
    cmap_Re, norm_Re = add_color.create_cmap(
        cmap_name=custom_cmap,
        vmin=3.1,
        vmax=3.7,
        # cmin = 0.1,
        cmax=0.7,
    )

    model_colour = cmap_Re(0.7)  # "blueviolet"
    guide_colour = cmap_Re(0.7)  # "blueviolet"

    coords_to_fit = []
    for sim_suite, sim_data in all_results.items():
        print("Looking at:", sim_suite)

        gamma_nl_stats = sim_data["fit_summaries"][MODEL_TYPE][BINNING_TYPE]["gamma_nl"]
        if gamma_nl_stats["p50"] is None: continue

        log10_gamma_nl_p50 = numpy.log10(gamma_nl_stats["p50"])
        log10_gamma_nl_err_lower = log10_gamma_nl_p50 - numpy.log10(gamma_nl_stats["p16"])
        log10_gamma_nl_err_upper = numpy.log10(gamma_nl_stats["p84"]) - log10_gamma_nl_p50

        duration_stats = sim_data["fit_summaries"][MODEL_TYPE][BINNING_TYPE]["nl_duration"]
        if duration_stats["p50"] is None: continue

        log10_delta_t_p50 = numpy.log10(duration_stats["p50"])
        log10_delta_t_err_lower = log10_delta_t_p50 - numpy.log10(duration_stats["p16"])
        log10_delta_t_err_upper = numpy.log10(duration_stats["p84"]) - log10_delta_t_p50

        Mach_stats = sim_data["sim_params"]["Mach"]
        log10_Mach_p50 = numpy.log10(Mach_stats["p50"])
        log10_Mach_err_lower = log10_Mach_p50 - numpy.log10(Mach_stats["p16"])
        log10_Mach_err_upper = numpy.log10(Mach_stats["p84"]) - log10_Mach_p50

        t_turb = 0.5 / Mach_stats["p50"]  # ell_turb / u_turb

        Re_stats = sim_data["sim_params"]["Re"]
        Re_p50 = Re_stats["p50"]
        if Re_p50 < 1000: continue
        Re_color = cmap_Re(norm_Re(numpy.log10(Re_p50)))

        if "288" in sim_suite:
            marker = "o"
            zorder = 1
        elif "576" in sim_suite:
            marker = "s"
            zorder = 3
        elif "1152" in sim_suite:
            marker = "D"
            zorder = 5
        else:
            print("Could not determine resolution for:", sim_suite)
            continue

        log10_Mach_jiggle = 0.05 * random.uniform(-1, 1),
        log10_gamma_nl_jiggle = 0.05 * random.uniform(-1, 1)
        log10_delta_t_jiggle = 0.05 * random.uniform(-1, 1)
        x = log10_Mach_p50 + log10_Mach_jiggle
        y1 = log10_gamma_nl_p50 + log10_gamma_nl_jiggle
        axs[0].errorbar(
            x,
            y1,
            xerr=[
                [log10_Mach_err_lower],
                [log10_Mach_err_upper],
            ],
            yerr=[
                [log10_gamma_nl_err_lower],
                [log10_gamma_nl_err_upper],
            ],
            fmt=marker,
            markerfacecolor=Re_color,
            zorder=zorder,
            markeredgecolor="black",
            ecolor="black",
            markersize=10,
            linewidth=2,
            capsize=3,
        )
        y2 = log10_delta_t_p50 + log10_delta_t_jiggle - numpy.log10(t_turb)
        axs[1].errorbar(
            x,
            y2,
            xerr=[
                [log10_Mach_err_lower],
                [log10_Mach_err_upper],
            ],
            yerr=[
                [log10_delta_t_err_lower],
                [log10_delta_t_err_upper],
            ],
            fmt=marker,
            markerfacecolor=Re_color,
            zorder=zorder,
            markeredgecolor="black",
            ecolor="black",
            markersize=10,
            linewidth=2,
            capsize=3,
        )
        coords_to_fit.append((
            numpy.float64(x),
            numpy.float64(y1),
            y2,
            numpy.float64(log10_delta_t_err_lower),
        ), )

    axs[0].set_xticklabels([])
    axs[1].set_xlabel(r"$\log_{10}(\mathcal{M})$")
    axs[0].set_ylabel(r"$\log_{10}(\alpha_{\rm nl})$")
    axs[1].set_ylabel(r"$\log_{10}\big((t_{\rm sat} - t_{\rm nl}) / t_0\big)$")
    axs[0].set_xlim([x_min, x_max])
    axs[1].set_xlim([x_min, x_max])
    axs[0].set_ylim([y0_min, y0_max])
    axs[1].set_ylim([y1_min, y1_max])
    axs[0].axvline(x=0, color="black", linestyle=":", linewidth=1.5)
    axs[1].axvline(x=0, color="black", linestyle=":", linewidth=1.5)

    x_values = numpy.linspace(-2, 2, 100)
    ax0_bounds = (x_min, x_max, y0_min, y0_max)

    ## subsonic growth rate
    subsonic_fit_results = fit_data.fit_line_with_fixed_slope(
        x_values=[coord[0] for coord in coords_to_fit if coord[0] < 0],
        y_values=[coord[1] for coord in coords_to_fit if coord[0] < 0],
        slope=3,
    )
    subsonic_a1_ave = subsonic_fit_results["intercept"]["best"]
    subsonic_a1_std = subsonic_fit_results["intercept"]["std"]
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
    supersonic_fit_results = fit_data.fit_1d_linear_model(
        x_values=[coord[0] for coord in coords_to_fit if 0 < coord[0]],
        y_values=[coord[1] for coord in coords_to_fit if 0 < coord[0]],
    )
    supersonic_a0_ave = supersonic_fit_results["slope"]["best"]
    supersonic_a0_std = supersonic_fit_results["slope"]["std"]
    supersonic_a1_ave = supersonic_fit_results["intercept"]["best"]
    supersonic_a1_std = supersonic_fit_results["intercept"]["std"]
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
    duration_fit_results = fit_data.fit_line_with_fixed_slope(
        x_values=[coord[0] for coord in coords_to_fit],
        y_values=[coord[2] for coord in coords_to_fit],
        y_sigmas=[coord[3] for coord in coords_to_fit],
        slope=0,
    )
    duration_a1_ave = duration_fit_results["intercept"]["best"]
    duration_a1_std = duration_fit_results["intercept"]["std"]
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
        fontsize=20,
        x_alignment="left",
        y_alignment="center",
    )

    ## annotate reference models
    annotate_axis.overlay_curve(
      ax       = axs[0],
      x_values = x_values,
      y_values = numpy.log10(3/38 * 2) + 3 * x_values, # xu and lazarian
      color    = model_colour,
      linestyle       = ":",
      linewidth       = 1.75,
      alpha    = 1.0,
      zorder   = 1,
    )
    annotate_axis.overlay_curve(
      ax       = axs[0],
      x_values = x_values,
      y_values = numpy.log10(0.05 * 2) + 3 * x_values, # beresnyak
      color    = model_colour,
      linestyle       = (10, (10, 3, 1, 3, 1, 3)),
      linewidth       = 1.75,
      alpha    = 1.0,
      zorder   = 1,
    )
    annotate_axis.add_custom_legend(
        ax=axs[0],
        artists=[
            "--",
            "-",
        ],
        colors=[
            "black",
            "black",
        ],
        labels=[
            supersonic_label,
            subsonic_label + r"$\, \mathcal{M}^3$",
        ],
        marker_size=8,
        line_width=1.5,
        fontsize=16,
        text_color="k",
        position="lower right",
        anchor=(1.0, 0.025),
        num_cols=1,
        text_padding=0.5,
        label_spacing=0.625,
        enable_frame=True,
        face_color="white",
        edge_color="white",
        frame_alpha=0.75,
    )
    annotate_axis.add_custom_legend(
        ax=axs[0],
        artists=[
            ":",
            (10, (10, 3, 1, 3, 1, 3)),
        ],
        colors=[
            model_colour,
            model_colour,
        ],
        labels=[
            r"$(3/38) \, u_0^3 / \ell_0$",
            r"$0.05 \, u_0^3 / \ell_0$",
        ],
        marker_size=8,
        line_width=1.5,
        fontsize=16,
        text_color="k",
        position="upper left",
        anchor=(0.0, 0.99),
        num_cols=1,
        text_padding=0.5,
    )
    guide_x0 = 0.3
    guide_y0 = -3.75
    guide_length = 0.5
    guide_x_y6, guide_y_y6 = generate_line(
        x_start=guide_x0,
        y_start=guide_y0,
        slope=6,
        line_length=guide_length,
        domain_bounds=ax0_bounds,
        domain_aspect_ratio=6 / 4,
    )
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=guide_x_y6,
        y_values=guide_y_y6,
        color=guide_colour,
        linestyle="-",
        linewidth=1.75,
        alpha=1.0,
        zorder=1,
    )
    annotate_axis.add_text(
        ax=axs[0],
        x_pos=0.715,
        y_pos=0.575,
        label=r"$\mathcal{M}^{6}$",
        x_alignment="center",
        y_alignment="center",
        font_color=guide_colour,
        fontsize=20,
    )
    guide_x_y4, guide_y_y4 = generate_line(
        x_start=guide_x0,
        y_start=guide_y0,
        slope=4,
        line_length=guide_length,
        domain_bounds=ax0_bounds,
        domain_aspect_ratio=6 / 4,
    )
    annotate_axis.overlay_curve(
        ax=axs[0],
        x_values=guide_x_y4,
        y_values=guide_y_y4,
        color=guide_colour,
        linestyle="--",
        linewidth=1.85,
        alpha=1.0,
        zorder=1,
    )
    annotate_axis.add_text(
        ax=axs[0],
        x_pos=0.85,
        y_pos=0.475,
        label=r"$\mathcal{M}^{4}$",
        x_alignment="center",
        y_alignment="center",
        font_color=guide_colour,
        fontsize=20,
    )

    cbar = add_color.add_cbar_from_cmap(
        ax=axs[0],
        cmap=cmap_Re,
        norm=norm_Re,
        label=r"$\log_{10}(\mathrm{Re})$",
        side="top",
        cbar_padding=0.015,
        fontsize=24,
    )
    cbar_ticks = [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(f"{cbar_tick:.1f}" for cbar_tick in cbar_ticks)
    annotate_axis.add_custom_legend(
        ax=axs[1],
        artists=["o", "s", "D"],
        labels=[r"$288^3$", r"$576^3$", r"$1152^3$"],
        colors=["k"] * 3,
        marker_size=8,
        line_width=1.5,
        fontsize=16,
        text_color="k",
        position="upper left",
        anchor=(-0.05, 1.0),
        num_cols=3,
        text_padding=0.0,
        column_spacing=0.0,
    )
    script_dir = Path(__file__).parent
    plot_path = script_dir / "nl_scalings.pdf"
    plot_manager.save_figure(fig, plot_path)


if __name__ == "__main__":
    random.seed(5)
    main()

## } SCRIPT
