## ###############################################################
## DEPENDANCIES
## ###############################################################

import re
import sys
import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager
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
  sim_name = f"Mach{Mach_number}Re{Re_number}Pm{Pm_number}Nres{Nres_number}v{version_number}"
  time_values, magnetic_energy_values = read_vi_data.read_vi_data(
    directory    = sim_directory,
    dataset_name = "mag"
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
    "data" : {
      "time" : time_values[1:],
      "magnetic_energy" : magnetic_energy_values[1:],
    }
  }


## ###############################################################
## MAIN PROGRAM
## ###############################################################

def main():
  script_directory = io_manager.get_caller_directory() # not strictly necessary
  base_output_directory = io_manager.combine_file_path_parts([ script_directory, "..", "data" ])
  io_manager.init_directory(base_output_directory, verbose=False)
  sim_directories = sorted(Path("/scratch").glob("*/nk7952/R*/Mach*/Pm*/576*"))
  for sim_directory in sim_directories:
    data_dict = load_data(sim_directory)
    sim_name = data_dict["sim_name"]
    sim_output_directory = io_manager.combine_file_path_parts([ base_output_directory, sim_name ])
    io_manager.init_directory(sim_output_directory)
    plot_params = dict(color="red", ls="-", lw=1, zorder=5)
    fig, axs = plot_manager.create_figure(num_rows=2, share_x=True)
    axs[0].plot(data_dict["data"]["time"], data_dict["data"]["magnetic_energy"], **plot_params)
    axs[1].plot(data_dict["data"]["time"], numpy.log10(data_dict["data"]["magnetic_energy"]), **plot_params)
    axs[0].set_ylabel(r"$\mathrm{energy}$")
    axs[1].set_ylabel(r"$\log_{10}(\mathrm{energy})$")
    axs[1].set_xlabel(r"time")
    fig_file_name = f"dataset.png"
    fig_file_path = io_manager.combine_file_path_parts([ sim_output_directory, fig_file_name ])
    plot_manager.save_figure(fig, fig_file_path)
    json_file_name = f"dataset.json"
    json_file_path = io_manager.combine_file_path_parts([ sim_output_directory, json_file_name ])
    json_files.save_dict_to_json_file(json_file_path, data_dict, overwrite=True)
    print(" ")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit(0)


## END OF SCRIPT