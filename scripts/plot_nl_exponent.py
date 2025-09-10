## { SCRIPT

##
## === DEPENDENCIES ===
##

from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from jormi.ww_io import io_manager, json_files
from jormi.ww_plots import plot_manager, plot_styler, annotate_axis, add_color

##
## === MAIN PROGRAM ===
##


def main():
    ## define paths
    script_dir = Path(__file__).parent
    figures_dir = (script_dir / ".." / "figures").resolve()
    io_manager.init_directory(figures_dir)
    fig_path = figures_dir / "nl_exponent.pdf"
    dataset_dir = (script_dir / ".." / "datasets" / "summary.json").resolve()
    dataset = json_files.read_json_file_into_dict(dataset_dir)
    ## setup figure
    plot_styler.apply_theme_globally()
    fig, ax = plot_manager.create_figure()
    ## define custom colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        name="black-green",
        colors=["#68287d", "#d0a7c7", "#f2f0e0", "#d5e370", "#275b0e"],
        N=256,
    )
    cmap, norm = add_color.create_cmap(
        cmap_name=custom_cmap,
        vmin=1.0,
        vmax=2.0,
    )
    ## loop over and plot each ensemble-averaged simulation suite
    for suite_name, suite_stats in dataset.items():
        print("Looking at:", suite_name)
        ## extract measured stats
        log10_Mach = suite_stats["measured"]["log10_Mach"]
        log10_Re = suite_stats["measured"]["log10_Re"]
        p_nl = suite_stats["measured"]["p_nl"]
        ## tweak plot params
        color = cmap(norm(p_nl["p50"]))
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
            log10_Mach["p50"],
            log10_Re["p50"],
            xerr=[
                [log10_Mach["std_lo"]],
                [log10_Mach["std_hi"]],
            ],
            yerr=[
                [log10_Re["std_lo"]],
                [log10_Re["std_hi"]],
            ],
            fmt=marker,
            color=color,
            mec="black",
            markersize=10,
            capsize=3,
            zorder=zorder,
        )
    ## label and save
    ax.set_xlim([-1.5, 1.0])
    ax.set_ylim([2.95, 3.8])
    ax.axvline(x=0, ls="--", color="black", lw=1.5, zorder=5)
    ax.set_xlabel(r"$\log_{10}(\mathcal{M})$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{Re})$")
    cbar = add_color.add_cbar_from_cmap(
        ax=ax,
        cmap=cmap,
        norm=norm,
        label=r"$p_\mathrm{nl}$",
        side="top",
        fontsize=24,
    )
    cbar_ticks = [1.0, 1.25, 1.5, 1.75, 2.0]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(f"{cbar_tick}" for cbar_tick in cbar_ticks)
    annotate_axis.add_custom_legend(
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
    plot_manager.save_figure(fig, fig_path)


##
## === ENTRY POINT ===
##

if __name__ == "__main__":
    main()

## } SCRIPT
