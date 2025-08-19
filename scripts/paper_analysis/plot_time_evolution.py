## ###############################################################
## DEPENDANCIES
## ###############################################################

import sys
import numpy
from pathlib import Path
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager, add_color, add_annotations


## ###############################################################
## MAIN PROGRAM
## ###############################################################

def main():
  num_points = 10**3
  summary_path = Path("/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_ssd_nl/datasets_v2/summary_stats.json")
  all_results = json_files.read_json_file_into_dict(summary_path)

  fig, axs = plot_manager.create_figure(num_rows=2, share_x=True)
  ax_inset = add_annotations.add_inset_axis(
    ax           = axs[0],
    bounds       = (0.46, 0.065, 0.475, 0.5),
    x_label_side = "top",
    y_label_side = "left",
  )

  custom_cmap = LinearSegmentedColormap.from_list(
    name   = "white-brown",
    colors = ["#024f92", "#067bf1", "#ffffff", "#f65d25", "#A41409"],
    N      = 256
  )
  cmap_Mach, norm_Mach = add_color.create_cmap(
    cmap_name = custom_cmap,
    vmin = -1.0,
    vmid = 0,
    vmax = 1.0,
    cmin = 0.1,
    cmax = 0.9,
  )
  
  sim_paths_Nres576 = io_manager.ItemFilter(
    include_string = ["Mach", "Re1500", "Pm1", "Nres576"]
  ).filter(
    directory = "/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_ssd_nl/datasets_v2/"
  )

  sim_paths_Nres1152 = io_manager.ItemFilter(
    include_string = ["Mach", "Re1500", "Pm1", "Nres1152"]
  ).filter(
    directory = "/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_ssd_nl/datasets_v2/"
  )

  sim_paths = [
    sim_path
    for sim_path in sim_paths_Nres576
    if not any(
      Mach_str in str(sim_path)
      for Mach_str in ["Mach0.3", "Mach0.5", "Mach0.8"]
    )
  ]
  sim_paths.extend([
    sim_path
    for sim_path in sim_paths_Nres1152
  ])

  sim_collections = defaultdict(list)
  for sim_path in sim_paths:
    data_filepath = io_manager.combine_file_path_parts([ sim_path, "dataset.json" ])
    if not io_manager.does_file_exist(data_filepath):
      print(f"Missing dataset.json for: {sim_path}")
      continue
    if "Mach0.8" in str(sim_path): continue
    sim_instance = json_files.read_json_file_into_dict(data_filepath)
    sim_name     = sim_instance["sim_name"].split("v")[0]
    t_turb       = sim_instance["plasma_params"]["t_turb"]
    target_Mach  = sim_instance["plasma_params"]["target_Mach"]
    if target_Mach < 0.1: continue
    time_values  = numpy.array(sim_instance["measured_data"]["time_values"])
    Emag_values  = numpy.array(sim_instance["measured_data"]["magnetic_energy_values"])
    sim_collections[sim_name].append((
      t_turb,
      target_Mach,
      time_values,
      Emag_values,
    ))

  for sim_name, sim_instances in sim_collections.items():
    Emag_sat = all_results[sim_name]["fit_summaries"]["free"]["bin_per_t0"]["sat_energy"]["p50"]
    biggest_t_min  = numpy.max([
      numpy.min(sim_data[2])
      for sim_data in sim_instances
    ])
    smallest_t_max = numpy.min([
      numpy.max(sim_data[2])
      for sim_data in sim_instances
    ])
    interp_time_values = numpy.linspace(biggest_t_min, smallest_t_max, num_points)
    Emag_matrix = []
    for sim_instance in sim_instances:
      interp_time_values, interp_Emag_values = interpolate_data.interpolate_1d(
        x_values = sim_instance[2],
        y_values = sim_instance[3] / Emag_sat,
        x_interp = interp_time_values,
        kind     = "linear",
      )
      Emag_matrix.append(interp_Emag_values)
    Emag_matrix = numpy.array(Emag_matrix)
    Emag_p16_vals = numpy.percentile(Emag_matrix, 16, axis=0)
    Emag_p50_vals = numpy.percentile(Emag_matrix, 50, axis=0)
    Emag_p84_vals = numpy.percentile(Emag_matrix, 84, axis=0)
    t_turb = sim_instances[0][0]
    target_Mach = sim_instances[0][1]
    color = cmap_Mach(norm_Mach(numpy.log10(target_Mach)))
    axs[0].plot(
      interp_time_values,
      numpy.log10(Emag_p50_vals),
      color=color, markeredgewidth=0.2, zorder=target_Mach
    )
    axs[0].fill_between(
      interp_time_values,
      numpy.log10(Emag_p16_vals),
      numpy.log10(Emag_p84_vals),
      color=color, alpha=0.5, zorder=target_Mach
    )
    axs[1].plot(
      interp_time_values,
      Emag_p50_vals,
      color=color, markeredgewidth=0.2, zorder=target_Mach
    )
    axs[1].fill_between(
      interp_time_values,
      Emag_p16_vals,
      Emag_p84_vals,
      color=color, alpha=0.5, zorder=target_Mach
    )
    ax_inset.plot(
      interp_time_values / t_turb,
      Emag_p50_vals,
      color=color, zorder=1/target_Mach
    )
    ax_inset.fill_between(
      interp_time_values / t_turb,
      Emag_p16_vals,
      Emag_p84_vals,
      color=color, alpha=0.5, zorder=1/target_Mach
    )
  
  axs[0].axhline(y=0, ls=":", color="black", zorder=100)
  axs[1].axhline(y=1, ls=":", color="black", zorder=100)
  axs[0].set_ylabel(r"$\log_{10}(\mathrm{E_\mathrm{mag}} / \mathrm{E_\mathrm{mag, sat}})$")
  axs[1].set_ylabel(r"$E_\mathrm{mag} / \mathrm{E_\mathrm{mag, sat}}$")
  axs[1].set_xlabel(r"$t / t_\mathrm{sc}$")
  axs[0].set_xlim([0, 400])
  axs[1].set_xlim([0, 400])
  axs[0].set_ylim([-10.5, 1])
  axs[1].set_ylim([-0.025, 1.5])

  ax_inset.set_xlim([0, 200])
  ax_inset.set_ylim([0, 2])
  ax_inset.axhline(y=1, ls=":", color="black", zorder=100)
  ax_inset.set_ylabel(r"$E_\mathrm{mag} / \mathrm{E_\mathrm{mag, sat}}$")
  ax_inset.set_xlabel(r"$t / t_0$", labelpad=8)

  add_color.add_cbar_from_cmap(
    ax           = axs[0],
    cmap         = cmap_Mach,
    norm         = norm_Mach,
    label        = r"$\log_{10}(\mathcal{M})$",
    side         = "top",
    cbar_padding = 1e-2,
    fontsize     = 24,
  )
  script_dir = Path(__file__).parent
  plot_path = script_dir / "time_evolution.pdf"
  plot_manager.save_figure(fig, plot_path)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################

if __name__ == "__main__":
  main()
  sys.exit(0)


## .