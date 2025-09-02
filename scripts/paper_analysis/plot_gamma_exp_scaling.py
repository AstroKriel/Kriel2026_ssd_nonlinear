## { SCRIPT

import numpy
import random
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from jormi.ww_io import json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color

# MODEL_TYPE = "linear"
MODEL_TYPE = "free"

# BINNING_TYPE = "100bins"
BINNING_TYPE = "bin_per_t0"

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


def main():
    summary_path = Path(
        "/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_ssd_nl/datasets_v2/summary_stats.json",
    )
    all_results = json_files.read_json_file_into_dict(summary_path)

    fig, ax = plot_manager.create_figure()

    custom_cmap = LinearSegmentedColormap.from_list(
        name="white-brown",
        colors=["#024f92", "#067bf1", "#d4d4d4", "#f65d25", "#A41409"],
        N=256,
    )
    cmap_Mach, norm_Mach = add_color.create_cmap(
        cmap_name=custom_cmap,
        vmin=-1.0,
        vmid=0,
        vmax=1.0,
        # cmin = 0.1,
        # cmax = 0.9,
    )

    coords_to_fit = []
    for sim_suite, sim_data in all_results.items():
        print("Looking at:", sim_suite)

        gamma_exp_stats = sim_data["fit_summaries"][MODEL_TYPE][BINNING_TYPE]["gamma_exp"]
        if gamma_exp_stats["p50"] is None: continue

        Mach_stats = sim_data["sim_params"]["Mach"]
        Mach_p50 = Mach_stats["p50"]
        if Mach_p50 <= 0: continue
        log10_Mach = numpy.log10(Mach_p50)

        Re_stats = sim_data["sim_params"]["Re"]
        Re_p50 = Re_stats["p50"]
        log10_Re_p50 = numpy.log10(Re_p50)
        log10_Re_err_lower = log10_Re_p50 - numpy.log10(Re_stats["p16"])
        log10_Re_err_upper = numpy.log10(Re_stats["p84"]) - log10_Re_p50
        if log10_Re_p50 < 3: continue
        if log10_Re_p50 > 3.5 and "288" in sim_suite:
            continue

        scaled_gamma_exp_p50 = numpy.array(gamma_exp_stats["p50"]) / Mach_p50
        scaled_gamma_exp_normed_p50 = numpy.log10(scaled_gamma_exp_p50)
        scaled_gamma_exp_normed_err_lower = (
            numpy.log10(scaled_gamma_exp_p50) - numpy.log10(numpy.array(gamma_exp_stats["p16"]) / Mach_p50)
        )
        scaled_gamma_exp_normed_err_upper = (
            numpy.log10(numpy.array(gamma_exp_stats["p84"]) / Mach_p50) - numpy.log10(scaled_gamma_exp_p50)
        )

        Mach_color = cmap_Mach(norm_Mach(log10_Mach))

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

        log10_Re_jiggle = 0.05 * random.uniform(-1, 1)
        ax.errorbar(
            log10_Re_p50 + log10_Re_jiggle,
            scaled_gamma_exp_normed_p50,
            xerr=[
                [log10_Re_err_lower],
                [log10_Re_err_upper],
            ],
            yerr=[
                [scaled_gamma_exp_normed_err_lower],
                [scaled_gamma_exp_normed_err_upper],
            ],
            fmt=marker,
            markerfacecolor=Mach_color,
            zorder=zorder,
            markeredgecolor="black",
            ecolor="black",
            markersize=10,
            lw=2,
            capsize=3,
        )
        coords_to_fit.append(
            (
                numpy.float64(log10_Re_p50 + log10_Re_jiggle),
                numpy.float64(scaled_gamma_exp_normed_p50),
                log10_Mach,
            ),
        )

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    rotation_bounds = (x_min, x_max, y_min, y_max)
    x_values = numpy.linspace(3, 4, 100)

    ## anotate subsonic fit
    subsonic_fit_results = fit_data.fit_line_with_fixed_slope(
        x_values=[coord[0] for coord in coords_to_fit if coord[2] < 0],
        y_values=[coord[1] for coord in coords_to_fit if coord[2] < 0],
        slope=1 / 2,
    )
    subsonic_a1_ave = subsonic_fit_results["intercept"]["best"]
    subsonic_a1_std = subsonic_fit_results["intercept"]["std"]
    annotate_axis.overlay_curve(
        ax=ax,
        x_values=x_values,
        y_values=(1 / 2) * x_values + subsonic_a1_ave,
        ls="-",
        lw=1.5,
    )
    subsonic_label = format_fit_label(
        intercept_best=subsonic_a1_ave,
        intercept_std=subsonic_a1_std,
        decimals=2,
    )
    subsonic_rotation = fit_data.get_line_angle(
        slope=1 / 2,
        domain_bounds=rotation_bounds,
        domain_aspect_ratio=6 / 4,
    )
    add_annotations.add_text(
        ax=ax,
        x_pos=0.585,
        y_pos=0.665,
        label=subsonic_label + r"$\, \mathrm{Re}^{1/2}$",
        x_alignment="center",
        y_alignment="center",
        rotate_deg=subsonic_rotation,
        font_color=cmap_Mach(norm_Mach(-1)),
    )

    ## anotate supersonic fit
    supersonic_fit_results = fit_data.fit_line_with_fixed_slope(
        x_values=[coord[0] for coord in coords_to_fit if 0 < coord[2]],
        y_values=[coord[1] for coord in coords_to_fit if 0 < coord[2]],
        slope=1 / 3,
    )
    supersonic_a1_ave = supersonic_fit_results["intercept"]["best"]
    supersonic_a1_std = supersonic_fit_results["intercept"]["std"]
    annotate_axis.overlay_curve(
        ax=ax,
        x_values=x_values,
        y_values=(1 / 3) * x_values + supersonic_a1_ave,
        ls="--",
        lw=1.5,
    )
    supersonic_label = format_fit_label(
        intercept_best=supersonic_a1_ave,
        intercept_std=supersonic_a1_std,
        decimals=2,
    )
    supersonic_rotation = fit_data.get_line_angle(
        slope=1 / 3.5,
        domain_bounds=rotation_bounds,
        domain_aspect_ratio=6 / 4,
    )
    add_annotations.add_text(
        ax=ax,
        x_pos=0.665,
        y_pos=0.15,
        label=supersonic_label + r"$\, \mathrm{Re}^{1/3}$",
        x_alignment="center",
        y_alignment="center",
        rotate_deg=supersonic_rotation,
        font_color=cmap_Mach(norm_Mach(1)),
    )

    x_ticks = [3.1, 3.3, 3.5, 3.7]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(f"{x_tick:.1f}" for x_tick in x_ticks)
    ax.set_xlabel(r"$\log_{10}(\mathrm{Re})$")
    ax.set_ylabel(r"$\log_{10}(\gamma_\mathrm{exp} \,t_0)$")
    add_color.add_cbar_from_cmap(
        ax=ax,
        cmap=cmap_Mach,
        norm=norm_Mach,
        label=r"$\log_{10}(\mathcal{M})$",
        side="top",
        fontsize=24,
    )
    add_annotations.add_custom_legend(
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
        fontsize=16,
        text_color="k",
        position="upper left",
        anchor=(-0.035, 1.0),
        num_cols=2,
        text_padding=0.0,
    )
    script_dir = Path(__file__).parent
    plot_path = script_dir / "gamma_exp_scaling.pdf"
    plot_manager.save_figure(fig, plot_path)


if __name__ == "__main__":
    random.seed(4)
    main()

## } SCRIPT
