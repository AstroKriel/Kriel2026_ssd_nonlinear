## { SCRIPT

import numpy
import random
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from jormi.ww_io import json_files
from jormi.ww_plots import plot_manager, add_annotations, add_color

# BINNING_TYPE = "100bins"
BINNING_TYPE = "bin_per_t0"


def extract_key_param_samples(fitted_posterior_samples):
    nl_exponent_samples = fitted_posterior_samples[:, 5]
    return nl_exponent_samples


def main():
    summary_path = Path(
        "/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_ssd_nl/datasets_v2/summary_stats.json",
    )
    all_results = json_files.read_json_file_into_dict(summary_path)

    fig, ax = plot_manager.create_figure()

    custom_cmap = LinearSegmentedColormap.from_list(
        name="black-green",
        colors=["#68287d", "#d0a7c7", "#f2f0e0", "#d5e370", "#275b0e"],
        N=256,
    )
    cmap_nl_exponent, norm_nl_exponent = add_color.create_cmap(
        cmap_name=custom_cmap,
        vmin=1.0,
        vmax=2.0,
        # cmin = 0.1,
        # cmax = 0.9,
    )

    for sim_suite, sim_data in all_results.items():
        print("Looking at:", sim_suite)

        nl_exponent_p50 = sim_data["fit_summaries"]["free"][BINNING_TYPE]["nl_exponent"]["p50"]
        if nl_exponent_p50 is None: continue

        Mach_stats = sim_data["sim_params"]["Mach"]
        log10_Mach_p50 = numpy.log10(Mach_stats["p50"])
        log10_Mach_err_lower = log10_Mach_p50 - numpy.log10(Mach_stats["p16"])
        log10_Mach_err_upper = numpy.log10(Mach_stats["p84"]) - log10_Mach_p50

        Re_stats = sim_data["sim_params"]["Re"]
        log10_Re_p50 = numpy.log10(Re_stats["p50"])
        log10_Re_err_lower = log10_Re_p50 - numpy.log10(Re_stats["p16"])
        log10_Re_err_upper = numpy.log10(Re_stats["p84"]) - log10_Re_p50

        nl_exponent_color = cmap_nl_exponent(norm_nl_exponent(nl_exponent_p50))

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

        ax.errorbar(
            log10_Mach_p50 + 0.05 * random.uniform(-1, 1),
            log10_Re_p50 + 0.05 * random.uniform(-1, 1),
            xerr=[
                [log10_Mach_err_lower],
                [log10_Mach_err_upper],
            ],
            yerr=[
                [log10_Re_err_lower],
                [log10_Re_err_upper],
            ],
            fmt=marker,
            color=nl_exponent_color,
            mec="black",
            markersize=10,
            capsize=3,
            zorder=zorder,
        )

    ax.set_xlim([-1.5, 1.0])
    ax.set_ylim([2.95, 3.8])
    ax.axvline(x=0, ls="--", color="black", lw=1.5, zorder=5)
    ax.set_xlabel(r"$\log_{10}(\mathcal{M})$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{Re})$")

    cbar = add_color.add_cbar_from_cmap(
        ax=ax,
        cmap=cmap_nl_exponent,
        norm=norm_nl_exponent,
        label=r"$p_\mathrm{nl}$",
        side="top",
        fontsize=24,
    )
    cbar_ticks = [1.0, 1.25, 1.5, 1.75, 2.0]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(f"{cbar_tick}" for cbar_tick in cbar_ticks)
    add_annotations.add_custom_legend(
        ax=ax,
        artists=["o", "s", "D"],
        labels=[r"$288^3$", r"$576^3$", r"$1152^3$"],
        colors=["k", "k", "k"],
        marker_size=8,
        line_width=1.5,
        fontsize=16,
        text_color="k",
        position="upper left",
        anchor=(0.0, 0.95),
    )
    script_dir = Path(__file__).parent
    plot_path = script_dir / "nl_exponent.pdf"
    plot_manager.save_figure(fig, plot_path)


if __name__ == "__main__":
    random.seed(4)
    main()

## } SCRIPT
