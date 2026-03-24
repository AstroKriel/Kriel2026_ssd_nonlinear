## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import sys
from pathlib import Path
from collections import defaultdict

## third-party
import numpy

## personal
from jormi import ww_lists
from jormi.ww_io import manage_io, json_io
from jormi.ww_data import interpolate_series
from jormi.ww_plots import manage_plots, style_plots, add_color
from jormi.ww_data.series_types import DataSeries
from jormi.ww_plots.color_palettes import DivergingPalette

##
## === MAIN PROGRAM
##


def main() -> None:
    num_points = 10**3
    script_dir = Path(__file__).parent
    figures_dir = (script_dir / ".." / ".." / "figures").resolve()
    manage_io.init_directory(figures_dir)
    datasets_dir = (script_dir / ".." / ".." / "datasets").resolve()
    summary_path = datasets_dir / "summary_stats.json"
    all_results = json_io.read_json_file_into_dict(summary_path)

    style_plots.set_theme()
    fig, axs = manage_plots.create_figure(num_rows=2, num_cols=1, share_x=True)
    axs = axs[:, 0]
    ax_inset = manage_plots.add_inset_axis(
        ax=axs[0],
        bounds=(0.45, 0.1, 0.475, 0.5),
        x_label_alignment="top",
        y_label_alignment="left",
    )

    palette_Mach = DivergingPalette.from_name(
        palette_name="blue-white-red",
        value_range=(-1.0, 1.0),
        mid_value=0.0,
        palette_range=(0.1, 0.9),
    )

    sim_paths_Nres576 = manage_io.ItemFilter(
        req_include_words=["Mach", "Re1500", "Pm1", "Nres576"],
    ).filter(
        directory=datasets_dir / "sims",
    )

    sim_paths_Nres1152 = manage_io.ItemFilter(
        req_include_words=["Mach", "Re1500", "Pm1", "Nres1152"],
    ).filter(
        directory=datasets_dir / "sims",
    )

    sim_paths = [
        sim_path for sim_path in sim_paths_Nres576
        if not any(Mach_str in str(sim_path) for Mach_str in ["Mach0.3", "Mach0.5", "Mach0.8"])
    ]
    sim_paths.extend([sim_path for sim_path in sim_paths_Nres1152])

    sim_collections = defaultdict(list)
    for sim_path in sim_paths:
        data_filepath = manage_io.combine_file_path_parts([sim_path, "sim_data.json"])
        if not manage_io.does_file_exist(data_filepath):
            print(f"Missing sim_data.json for: {sim_path}")
            continue
        sim_instance = json_io.read_json_file_into_dict(data_filepath)
        sim_name = sim_instance["details"]["name"].split("v")[0]
        t_turb = sim_instance["details"]["t_0"]
        target_Mach = sim_instance["details"]["target_Mach"]
        if target_Mach < 0.1: continue
        time_values = numpy.array(sim_instance["time_series"]["time"])
        Emag_values = numpy.array(sim_instance["time_series"]["Emag"])
        sim_collections[sim_name].append((
            t_turb,
            target_Mach,
            time_values,
            Emag_values,
        ))

    for sim_name, sim_instances in sim_collections.items():
        Emag_sat = all_results[sim_name]["fit_summaries"]["free"]["bin_per_t0"]["sat_energy"]["p50"]
        biggest_t_min = numpy.max([numpy.min(sim_data[2]) for sim_data in sim_instances])
        smallest_t_max = numpy.min([numpy.max(sim_data[2]) for sim_data in sim_instances])
        interp_time_values = numpy.linspace(biggest_t_min, smallest_t_max, num_points)
        Emag_matrix_list = []
        for sim_instance in sim_instances:
            interp_result = interpolate_series.interpolate_1d(
                DataSeries(
                    x_values=sim_instance[2],
                    y_values=sim_instance[3] / Emag_sat,
                ),
                x_interp=interp_time_values,
                spline_order=1,
            )
            interp_time_values, interp_Emag_values = interp_result.x_values, interp_result.y_values
            Emag_matrix_list.append(interp_Emag_values)
        Emag_matrix = numpy.array(Emag_matrix_list)
        Emag_p16_vals = numpy.percentile(Emag_matrix, 16, axis=0)
        Emag_p50_vals = numpy.percentile(Emag_matrix, 50, axis=0)
        Emag_p84_vals = numpy.percentile(Emag_matrix, 84, axis=0)
        t_turb = sim_instances[0][0]
        target_Mach = sim_instances[0][1]
        color = palette_Mach.mpl_cmap(palette_Mach.mpl_norm(numpy.log10(target_Mach)))
        index_start = ww_lists.get_index_of_closest_value(
            values=numpy.log10(Emag_p50_vals),
            target=-10,
        )
        axs[0].plot(
            (interp_time_values - interp_time_values[index_start]) / t_turb,
            numpy.log10(Emag_p50_vals),
            color=color,
            markeredgewidth=0.2,
            zorder=1 / target_Mach,
        )
        axs[0].fill_between(
            (interp_time_values - interp_time_values[index_start]) / t_turb,
            numpy.log10(Emag_p16_vals),
            numpy.log10(Emag_p84_vals),
            color=color,
            alpha=0.35,
            zorder=1 / target_Mach,
        )
        axs[1].plot(
            (interp_time_values - interp_time_values[index_start]) / t_turb,
            Emag_p50_vals,
            color=color,
            markeredgewidth=0.2,
            zorder=1 / target_Mach,
        )
        axs[1].fill_between(
            (interp_time_values - interp_time_values[index_start]) / t_turb,
            Emag_p16_vals,
            Emag_p84_vals,
            color=color,
            alpha=0.35,
            zorder=1 / target_Mach,
        )
        ax_inset.plot(
            numpy.log10(interp_time_values - interp_time_values[index_start]),
            Emag_p50_vals,
            color=color,
            zorder=1 / target_Mach,
        )
        ax_inset.fill_between(
            numpy.log10(interp_time_values - interp_time_values[index_start]),
            Emag_p16_vals,
            Emag_p84_vals,
            color=color,
            alpha=0.35,
            zorder=1 / target_Mach,
        )

    axs[0].axhline(y=0, ls=":", color="black", zorder=100)
    axs[1].axhline(y=1, ls=":", color="black", zorder=100)
    axs[0].set_ylabel(r"$\log_{10}(\mathrm{E_\mathrm{mag}} / \mathrm{E_\mathrm{mag, sat}})$")
    axs[1].set_ylabel(r"$E_\mathrm{mag} / \mathrm{E_\mathrm{mag, sat}}$")
    axs[1].set_xlabel(r"$t / t_0$")
    axs[0].set_xlim([0, 200])
    axs[1].set_xlim([0, 200])
    axs[0].set_ylim([-10.5, 1])
    axs[1].set_ylim([-0.025, 1.5])

    ax_inset.set_xlim((0.5, 3))
    ax_inset.set_ylim((0, 2))
    ticks = [1, 2, 3]
    ax_inset.set_xticks(ticks)
    ax_inset.set_xticklabels(f"{tick}" for tick in ticks)
    ax_inset.axhline(y=1, ls=":", color="black", zorder=100)
    ax_inset.set_ylabel(r"$E_\mathrm{mag} / \mathrm{E_\mathrm{mag, sat}}$")
    ax_inset.set_xlabel(r"$\log_{10}(t / t_\mathrm{sc})$", labelpad=8)

    add_color.add_colorbar(
        ax=axs[0],
        palette=palette_Mach,
        label=r"$\log_{10}(\mathcal{M})$",
        cbar_side="top",
        cbar_pad=1e-2,
        label_size=24,
    )
    fig_path = figures_dir / "time_evolution.pdf"
    manage_plots.save_figure(fig, fig_path)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()
    sys.exit(0)

## } SCRIPT
