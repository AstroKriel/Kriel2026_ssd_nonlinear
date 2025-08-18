## ###############################################################
## DEPENDANCIES
## ###############################################################

import re
import sys
import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_plots import plot_manager
from ww_flash_sims.sim_io import read_vi_data


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################

## extract target simulation parameters from directory path
## assumes naming convention with Re/Mach/Pm/resolution(version)
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

## load and store both desired simulation parameters and volume integrated quantities
def load_data(sim_directory: str | Path):
  Mach_number, Re_number, Pm_number, Nres_number, version_number = extract_sim_params(sim_directory)
  sim_name = f"Mach{Mach_number}Re{Re_number}Pm{Pm_number}Nres{Nres_number}v{version_number}"
  time_values, magnetic_energy_values = read_vi_data.read_vi_data(directory=sim_directory, dataset_name="mag")
  _, kinetic_energy_values = read_vi_data.read_vi_data(directory=sim_directory, dataset_name="kin")
  _, rms_Mach_values = read_vi_data.read_vi_data(directory=sim_directory, dataset_name="Mach")
  return {
    "sim_name" : sim_name,
    "sim_directory" : str(sim_directory),
    "plasma_params" : {
      "target_Mach" : Mach_number,
      "target_Re" : Re_number,
      "target_Pm" : Pm_number,
      "resolution" : Nres_number,
      "version" : version_number,
      "t_turb" : 0.5 / Mach_number, # ell_turb / u_turb
    },
    "measured_data" : {
      "time_values" : time_values[1:],
      "rms_Mach_values" : rms_Mach_values[1:],
      "magnetic_energy_values" : magnetic_energy_values[1:],
      "kinetic_energy_values" : kinetic_energy_values[1:]
    }
  }

## generate diagnostic plots and save dataset
def plot_and_save_data(dataset: dict, output_directory: Path):
  io_manager.init_directory(output_directory)
  fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
  axs[0].plot(dataset["measured_data"]["time_values"], dataset["measured_data"]["rms_Mach_values"], color="blue")
  axs[1].plot(dataset["measured_data"]["time_values"], dataset["measured_data"]["magnetic_energy_values"], color="red")
  axs[2].plot(dataset["measured_data"]["time_values"], numpy.log10(dataset["measured_data"]["magnetic_energy_values"]), color="red")
  axs[2].plot(dataset["measured_data"]["time_values"], numpy.log10(dataset["measured_data"]["kinetic_energy_values"]), color="blue")
  axs[0].set_ylabel(r"$\mathcal{M}$")
  axs[1].set_ylabel(r"$\mathrm{energy}$")
  axs[2].set_ylabel(r"$\log_{10}(\mathrm{energy})$")
  axs[2].set_xlabel("time")
  plot_path = io_manager.combine_file_path_parts([ output_directory, "dataset.png" ])
  json_path = io_manager.combine_file_path_parts([ output_directory, "dataset.json" ])
  plot_manager.save_figure(fig, plot_path)
  json_files.save_dict_to_json_file(json_path, dataset, overwrite=True)


## ###############################################################
## MAIN PROGRAM
## ###############################################################

def main():
  base_output_directory = io_manager.combine_file_path_parts([ "/scratch/jh2/nk7952/ssd_sims" ])
  io_manager.init_directory(base_output_directory, verbose=False)
  ## find matching simulation directories under /scratch/
  sim_directories = [
    sim_directory
    for sim_directory in sorted(Path("/scratch/").glob("*/nk7952/R*/Mach*/Pm*/*"))
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
  ## display which directories are going to be looked at
  [
    print(sim_directory)
    for sim_directory in sim_directories
  ]
  print(" ")
  ## process each simulation
  for sim_directory in sim_directories:
    dataset = load_data(sim_directory)
    sim_output_directory = io_manager.combine_file_path_parts([ base_output_directory, dataset["sim_name"] ])
    plot_and_save_data(dataset, sim_output_directory)
    print(" ")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################

if __name__ == "__main__":
  main()
  sys.exit(0)


## END OF SCRIPT