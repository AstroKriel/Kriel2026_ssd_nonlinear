## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party

## personal
from jormi.ww_io import manage_io, json_io
from jormi.ww_plots import manage_plots, style_plots, annotate_axis, add_color
from jormi.ww_plots.color_palettes import SequentialPalette

##
## === MAIN PROGRAM
##


def main() -> None:
    ## define paths
    script_dir = Path(__file__).parent
    figures_dir = (script_dir / ".." / ".." / "figures").resolve()
    manage_io.init_directory(figures_dir)
    fig_path = figures_dir / "nl_exponent.pdf"
    dataset_path = (script_dir / ".." / ".." / "datasets" / "summary.json").resolve()
    dataset = json_io.read_json_file_into_dict(dataset_path)
    ## setup figure
    style_plots.set_theme()
    fig, ax = manage_plots.create_figure()
    ## define custom colormap
    palette = SequentialPalette.from_name(
        palette_name="purple-white-green",
        value_range=(1.0, 2.0),
    )
    ## loop over and plot each ensemble-averaged simulation suite
    for suite_name, suite_stats in dataset.items():
        print("Looking at:", suite_name)
        sim_path = (script_dir / ".." / ".." / "datasets" / "sims" / f"{suite_name}v1" / "sim_data.json").resolve()
        sim_data = json_io.read_json_file_into_dict(sim_path)
        target_Mach = sim_data["details"]["target_Mach"]
        target_Re = sim_data["details"]["target_Re"]
        dataset[suite_name]["input"]["target_Mach"] = target_Mach
        dataset[suite_name]["input"]["target_Re"] = target_Re
        ## extract measured stats
        log10_Mach = suite_stats["measured"]["log10_Mach"]
        log10_Re = suite_stats["measured"]["log10_Re"]
        p_nl = suite_stats["measured"]["p_nl"]
        ## tweak plot params
        color = palette.mpl_cmap(palette.mpl_norm(p_nl["p50"]))
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
    json_io.save_dict_to_json_file(str(dataset_path).replace(".json", "_v2.json"), dataset)
    ## label and save
    ax.set_xlim((-1.5, 1.0))
    ax.set_ylim((2.95, 3.8))
    ax.axvline(x=0, ls="--", color="black", lw=1.5, zorder=5)
    ax.set_xlabel(r"$\log_{10}(\mathcal{M})$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{Re})$")
    cbar = add_color.add_colorbar(
        ax=ax,
        palette=palette,
        label=r"$p_\mathrm{nl}$",
        cbar_side="top",
        label_size=24,
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
        text_size=16,
        text_color="k",
        position="upper left",
        anchor=(0.0, 0.95),
    )
    manage_plots.save_figure(fig, fig_path)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
