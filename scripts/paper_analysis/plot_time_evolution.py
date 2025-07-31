## ###############################################################
## DEPENDANCIES
## ###############################################################

import re
import sys
import numpy
from pathlib import Path
from collections import defaultdict
from jormi.utils import list_utils
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager, add_color, add_annotations
from ww_flash_sims.sim_io import read_vi_data


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################

def extract_sim_params(sim_directory: str | Path):
  sim_directory = str(sim_directory)
  match_plasma_pattern = re.search(r"Re(\d+)/Mach([\d.]+)/Pm(\d+)", sim_directory)
  if not match_plasma_pattern: raise ValueError(f"Could not extract plasma parameters from path: {sim_directory}")
  Mach_number = float(match_plasma_pattern.group(2))
  Re_number   = int(match_plasma_pattern.group(1))
  Pm_number   = int(match_plasma_pattern.group(3))
  match_sim_pattern = re.search(r"/(\d+)(?:v(\d+))?/?$", sim_directory)
  if not match_sim_pattern: raise ValueError(f"Could not extract resolution from path: {sim_directory}")
  Nres_number    = int(match_sim_pattern.group(1))
  version_number = int(match_sim_pattern.group(2)) if match_sim_pattern.group(2) else 1
  return Mach_number, Re_number, Pm_number, Nres_number, version_number

def load_data(sim_directory: str | Path):
  Mach_number, Re_number, Pm_number, Nres_number, version_number = extract_sim_params(sim_directory)
  sim_name = f"Mach{Mach_number}Re{Re_number}Pm{Pm_number}"
  time_values, magnetic_energy_values = read_vi_data.read_vi_data(
    directory    = sim_directory,
    dataset_name = "mag"
  )
  _, kinetic_energy_values = read_vi_data.read_vi_data(
    directory    = sim_directory,
    dataset_name = "kin"
  )
  _, Mach_values  = read_vi_data.read_vi_data(
    directory    = sim_directory,
    dataset_name = "Mach"
  )
  # energy_ratio_values = numpy.array(magnetic_energy_values) / numpy.array(kinetic_energy_values)
  energy_ratio_values = numpy.array(magnetic_energy_values) #/ numpy.mean(magnetic_energy_values[len(time_values)//2 : 3*len(time_values)//4])
  start_index = list_utils.get_index_of_closest_value(
    values = numpy.log10(energy_ratio_values),
    target = -10,
  )
  return {
    "sim_name" : sim_name,
    "sim_directory" : str(sim_directory),
    "plasma_params" : {
      "t_turb" : 0.5 / Mach_number, # ell_turb / u_turb
      "Mach" : Mach_number,
      "Re" : Re_number,
      "Pm" : Pm_number,
    },
    "raw_data" : {
      "time" : numpy.array(time_values[start_index:]) - time_values[start_index],
      "Mach_values" : Mach_values[start_index:],
      "kinetic_energy" : kinetic_energy_values[start_index:],
      "magnetic_energy" : magnetic_energy_values[start_index:],
      "energy_ratio" : energy_ratio_values[start_index:],
      # "energy_ratio" : energy_ratio_values[start_index:],
    }
  }


## ###############################################################
## MAIN PROGRAM
## ###############################################################

def main():
  base_output_directory = io_manager.combine_file_path_parts(["/scratch/jh2/nk7952/kriel2025_nl_data"])
  io_manager.init_directory(base_output_directory, verbose=False)

  Re_varied_directories = sorted(Path("/scratch/").glob("*/nk7952/Re*/Mach0.5/Pm1/576"))
  Mach_varied_directories = sorted(Path("/scratch/").glob("*/nk7952/Re1500/Mach*/Pm1/*"))
  Mach_varied_directories = [
    sim_directory
    for sim_directory in Mach_varied_directories
    if io_manager.does_file_exist(
      directory   = sim_directory,
      file_name   = "Turb.dat",
      raise_error = False
    ) and (
      any(
        sim_nres in str(sim_directory)
        for sim_nres in ["288", "576", "1152"]
      )
      and
      "anti" not in str(sim_directory)
    )
  ]

  fig, axs = plot_manager.create_figure(num_rows=2, share_x=True)
  ax_inset = add_annotations.add_inset_axis(
    ax           = axs[0],
    bounds       = (0.45, 0.065, 0.525, 0.5),
    x_label_side = "top",
    y_label_side = "left",
  )
  cmap_Mach, norm_Mach = add_color.create_cmap(
    cmap_name="cmr.watermelon",
    vmin=numpy.log10(0.1), vmid=0, vmax=numpy.log10(5)
  )
  cmap_Re, norm_Re = add_color.create_cmap(
    cmap_name = "Blues",
    vmin      = numpy.log10(100),
    vmax      = numpy.log10(5000),
  )

  Mach_varied_data = defaultdict(list)
  scaled_Mach_varied_data = defaultdict(list)
  for sim_directory in Mach_varied_directories:
    data_dict = load_data(sim_directory)
    sim_name = data_dict["sim_name"]
    t_turb = data_dict["plasma_params"]["t_turb"]
    time_vals = data_dict["raw_data"]["time"]
    energy_ratio = data_dict["raw_data"]["energy_ratio"]
    Mach_varied_data[sim_name].append((
      time_vals,
      energy_ratio,
      data_dict["plasma_params"]["Mach"]
    ))
    scaled_Mach_varied_data[sim_name].append((
      time_vals / t_turb,
      energy_ratio,
      data_dict["plasma_params"]["Mach"]
    ))
  for sim_name, simulations in Mach_varied_data.items():
    t_min = max(sim[0][0] for sim in simulations)
    t_max = min(sim[0][-1] for sim in simulations)
    t_common = numpy.linspace(t_min, t_max, 500)
    energy_matrix = []
    for t, energy, _ in simulations:
      t_common, interp = interpolate_data.interpolate_1d(
        x_values = t,
        y_values = energy,
        x_interp = t_common,
        kind     = "linear",
      )
      energy_matrix.append(interp)
    energy_matrix = numpy.array(energy_matrix)
    median_vals = numpy.median(energy_matrix, axis=0)
    low_vals = numpy.percentile(energy_matrix, 25, axis=0)
    high_vals = numpy.percentile(energy_matrix, 75, axis=0)
    Mach_number = simulations[0][2]
    color = cmap_Mach(norm_Mach(numpy.log10(Mach_number)))
    axs[0].plot(t_common, numpy.log10(median_vals), color=color, lw=2)
    axs[0].fill_between(t_common, numpy.log10(low_vals), numpy.log10(high_vals), color=color, alpha=0.5)
    axs[1].plot(t_common, median_vals, color=color, lw=2)
    axs[1].fill_between(t_common, low_vals, high_vals, color=color, alpha=0.5)
  for sim_name, simulations in scaled_Mach_varied_data.items():
    t_min = max(sim[0][0] for sim in simulations)
    t_max = min(sim[0][-1] for sim in simulations)
    t_common = numpy.linspace(t_min, t_max, 500)
    energy_matrix = []
    for t, energy, _ in simulations:
      t_common, interp = interpolate_data.interpolate_1d(
        x_values = t,
        y_values = energy,
        x_interp = t_common,
        kind     = "linear",
      )
      energy_matrix.append(interp)
    energy_matrix = numpy.array(energy_matrix)
    median_vals = numpy.median(energy_matrix, axis=0)
    low_vals = numpy.percentile(energy_matrix, 25, axis=0)
    high_vals = numpy.percentile(energy_matrix, 75, axis=0)
    Mach_number = simulations[0][2]
    color = cmap_Mach(norm_Mach(numpy.log10(Mach_number)))
    ax_inset.plot(t_common, numpy.log10(median_vals), color=color, lw=2)
    ax_inset.fill_between(t_common, numpy.log10(low_vals), numpy.log10(high_vals), color=color, alpha=0.5)

  axs[0].axhline(y=0, ls=":", color="black")
  axs[1].axhline(y=1, ls=":", color="black")
  axs[0].set_ylabel(r"$\log_{10}(\mathrm{E_\mathrm{mag}} / \mathrm{E_\mathrm{mag, sat}})$")
  axs[1].set_ylabel(r"$E_\mathrm{mag} / \mathrm{E_\mathrm{mag, sat}}$")
  axs[1].set_xlabel(r"$t$")
  axs[0].set_xlim([0, 400])
  axs[1].set_xlim([0, 400])
  axs[0].set_ylim([-10.5, 1])
  axs[1].set_ylim([-0.025, 1.5])

  ax_inset.axhline(y=0, ls=":", color="black")
  ax_inset.set_ylabel(r"$E_\mathrm{mag} / \mathrm{E_\mathrm{mag, sat}}$")
  ax_inset.set_xlabel(r"$t / t_0$")

  add_color.add_cbar_from_cmap(
    ax    = axs[0],
    cmap  = cmap_Mach,
    norm  = norm_Mach,
    label = r"$\log_{10}(\mathcal{M})$",
    side  = "top",
    cbar_padding = 1e-2
  )

  plot_manager.save_figure(fig, "time_evolution.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################

if __name__ == "__main__":
  main()
  sys.exit(0)


## END OF SCRIPT