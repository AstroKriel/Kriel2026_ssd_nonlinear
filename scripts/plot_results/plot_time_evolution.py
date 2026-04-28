## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray

## personal
from jormi import ww_lists
from jormi.ww_io import manage_io, json_io
from jormi.ww_data import interpolate_series, series_types
from jormi.ww_plots import add_color, color_palettes, manage_plots
from jormi.ww_types import box_positions

## local
import plot_helpers

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class SimInstance:
    t_0: float
    target_Mach: float
    time_values: NDArray[Any]
    Emag_values: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        if len(self.time_values) != len(self.Emag_values):
            raise ValueError(
                f"time_values and Emag_values must have the same length, "
                f"got {len(self.time_values)} and {len(self.Emag_values)}",
            )


##
## === PIPELINE STAGES
##


def load_sim_collections(
    datasets_dir: Path,
) -> dict[str, list[SimInstance]]:
    sim_paths_Nres576 = manage_io.filter_directory(
        datasets_dir / "sims",
        req_include_words=["Mach", "Re1500", "Pm1", "Nres576"],
    )
    sim_paths_Nres1152 = manage_io.filter_directory(
        datasets_dir / "sims",
        req_include_words=["Mach", "Re1500", "Pm1", "Nres1152"],
    )
    sim_paths = [
        _sim_path for _sim_path in sim_paths_Nres576
        if not any(_exclude in str(_sim_path) for _exclude in ["Mach0.3", "Mach0.5", "Mach0.8"])
    ]
    sim_paths.extend(sim_paths_Nres1152)
    sim_collections: dict[str, list[SimInstance]] = defaultdict(list)
    for sim_path in sim_paths:
        data_filepath = manage_io.combine_file_path_parts([sim_path, "sim_data.json"])
        if not manage_io.does_file_exist(data_filepath):
            print(f"Missing sim_data.json for: {sim_path}")
            continue
        raw = json_io.read_json_file_into_dict(data_filepath)
        sim_name = raw["details"]["name"].split("v")[0]
        target_Mach = raw["details"]["target_Mach"]
        if target_Mach < 0.1:
            continue
        sim_collections[sim_name].append(
            SimInstance(
                t_0=raw["details"]["t_0"],
                target_Mach=target_Mach,
                time_values=numpy.array(raw["time_series"]["time"]),
                Emag_values=numpy.array(raw["time_series"]["Emag"]),
            ),
        )
    return dict(sim_collections)


def plot_series(
    *,
    axs: Any,
    ax_inset: Any,
    sim_collections: dict[str, list[SimInstance]],
    all_results: dict[str, Any],
    palette_Mach: color_palettes.DivergingPalette,
    num_points: int = 10**3,
) -> None:
    for sim_name, sim_instances in sim_collections.items():
        free_fits = all_results.get(sim_name, {}).get("fit_summaries", {}).get("free", {})
        if "bin_per_t0" not in free_fits:
            print(f"Skipping {sim_name}: no free fit data in summary_stats.json")
            continue
        Emag_sat = free_fits["bin_per_t0"]["sat_energy"]["p50"]
        biggest_t_min = numpy.max([numpy.min(_sim.time_values) for _sim in sim_instances])
        smallest_t_max = numpy.min([numpy.max(_sim.time_values) for _sim in sim_instances])
        interp_time_values = numpy.linspace(biggest_t_min, smallest_t_max, num_points)
        Emag_matrix_list = []
        for sim_instance in sim_instances:
            interp_result = interpolate_series.interpolate_1d(
                series_types.DataSeries(
                    x_values=sim_instance.time_values,
                    y_values=sim_instance.Emag_values / Emag_sat,
                ),
                x_interp=interp_time_values,
                spline_order=1,
            )
            interp_time_values, interp_Emag_values = interp_result.x_values, interp_result.y_values
            Emag_matrix_list.append(interp_Emag_values)
        Emag_matrix = numpy.array(Emag_matrix_list)
        Emag_p16 = numpy.percentile(Emag_matrix, 16, axis=0)
        Emag_p50 = numpy.percentile(Emag_matrix, 50, axis=0)
        Emag_p84 = numpy.percentile(Emag_matrix, 84, axis=0)
        t_turb = sim_instances[0].t_0
        target_Mach = sim_instances[0].target_Mach
        color = palette_Mach.mpl_cmap(
            palette_Mach.mpl_norm(
                numpy.log10(
                    target_Mach,
                ),
            ),
        )
        index_start = ww_lists.get_index_of_closest_value(
            values=list(
                numpy.log10(
                    Emag_p50,
                ),
            ),
            target=-10,
        )
        t_shifted = (interp_time_values - interp_time_values[index_start]) / t_turb
        axs[0].plot(
            t_shifted,
            numpy.log10(Emag_p50),
            color=color,
            markeredgewidth=0.2,
            zorder=1 / target_Mach,
        )
        axs[0].fill_between(
            t_shifted,
            numpy.log10(Emag_p16),
            numpy.log10(Emag_p84),
            color=color,
            alpha=0.35,
            zorder=1 / target_Mach,
        )
        axs[1].plot(
            t_shifted,
            Emag_p50,
            color=color,
            markeredgewidth=0.2,
            zorder=1 / target_Mach,
        )
        axs[1].fill_between(
            t_shifted,
            Emag_p16,
            Emag_p84,
            color=color,
            alpha=0.35,
            zorder=1 / target_Mach,
        )
        t_log = numpy.log10(interp_time_values - interp_time_values[index_start])
        ax_inset.plot(
            t_log,
            Emag_p50,
            color=color,
            zorder=1 / target_Mach,
        )
        ax_inset.fill_between(
            t_log,
            Emag_p16,
            Emag_p84,
            color=color,
            alpha=0.35,
            zorder=1 / target_Mach,
        )


def style_axes(
    *,
    axs: Any,
    ax_inset: Any,
    palette_Mach: color_palettes.DivergingPalette,
) -> None:
    add_color.add_colorbar(
        ax=axs[0],
        palette=palette_Mach,
        label=r"$\log_{10}(\mathcal{M})$",
        cbar_side=box_positions.Positions.Side.Top,
        cbar_pad=1e-2,
        label_size=24,
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
    ax_inset.set_xticklabels(f"{_tick}" for _tick in ticks)
    ax_inset.axhline(y=1, ls=":", color="black", zorder=100)


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
    all_results = json_io.read_json_file_into_dict(datasets_dir / "summary_stats.json")
    palette_Mach = color_palettes.DivergingPalette.from_name(
        palette_name="blue-white-red",
        value_range=(-1.0, 1.0),
        mid_value=0.0,
        palette_range=(0.1, 0.9),
    )
    sim_collections = load_sim_collections(datasets_dir)
    fig, axs = manage_plots.create_figure(
        num_rows=2,
        num_cols=1,
        share_x=True,
    )
    axs = axs[:, 0]
    ax_inset = manage_plots.add_inset_axis(
        ax=axs[0],
        bounds=(0.45, 0.1, 0.475, 0.5),
        x_label=r"$\log_{10}(t / t_\mathrm{sc})$",
        y_label=r"$E_\mathrm{mag} / \mathrm{E_\mathrm{mag, sat}}$",
        x_label_alignment=box_positions.Positions.Side.Top,
        y_label_alignment=box_positions.Positions.Side.Left,
    )
    plot_series(
        axs=axs,
        ax_inset=ax_inset,
        sim_collections=sim_collections,
        all_results=all_results,
        palette_Mach=palette_Mach,
    )
    style_axes(
        axs=axs,
        ax_inset=ax_inset,
        palette_Mach=palette_Mach,
    )
    manage_plots.save_figure(
        fig=fig,
        fig_path=figures_dir / "time_evolution.pdf",
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()
    sys.exit(0)

## } SCRIPT
