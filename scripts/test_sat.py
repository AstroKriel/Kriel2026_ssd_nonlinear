## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
import argparse
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from jormi.utils import list_utils
from jormi.ww_plots import plot_manager
from jormi.ww_io import io_manager, json_files
from my_mcmc_routine.mcmc_stage_1 import Stage1MCMCRoutine
from my_mcmc_routine.mcmc_stage_2_free import Stage2MCMCRoutine_free
from my_mcmc_routine.mcmc_stage_2_linear import Stage2MCMCRoutine_linear
from my_mcmc_routine.mcmc_stage_2_quadratic import Stage2MCMCRoutine_quadratic
from my_mcmc_routine import mcmc_utils
from my_mcmc_routine.plot_final_fits import PlotFinalFits


## ###############################################################
## PROGRAM MAIN
## ###############################################################

def main():
  sim_directory = "/scratch/jh2/nk7952/kriel2025_nl_data/Mach5.0Re1500Pm1Nres576v2"
  json_path = io_manager.combine_file_path_parts([sim_directory, "dataset.json"])
  sim_data = json_files.read_json_file_into_dict(json_path)
  raw_time = numpy.array(sim_data["raw_data"]["time"])
  raw_magnetic_energy = numpy.array(sim_data["raw_data"]["magnetic_energy"])
  binned_results = mcmc_utils.compute_binned_data(
    x_values = raw_time,
    y_values = raw_magnetic_energy,
    num_bins = 100
  )
  binned_time = binned_results["x_bin_centers"]
  binned_energy = binned_results["y_ave_s"]
  binned_log10_energy = binned_results["log10_y_ave_s"]
  energy_derivative = numpy.gradient(binned_energy, binned_time)
  log10_energy_derivative = numpy.gradient(binned_log10_energy, binned_time)
  smooth_energy_derivative = gaussian_filter1d(energy_derivative, sigma=2)
  smooth_log10_energy_derivative = gaussian_filter1d(log10_energy_derivative, sigma=2)
  fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
  axs[0].plot(binned_time, binned_energy, color="red", ls="-", lw=1.5)
  axs[1].plot(binned_time, energy_derivative, color="red", ls="-", lw=1.5)
  axs[2].plot(binned_time, log10_energy_derivative, color="red", ls="-", lw=1.5)
  axs[1].plot(binned_time, smooth_energy_derivative, color="blue", ls="-", lw=1.5)
  axs[2].plot(binned_time, smooth_log10_energy_derivative, color="blue", ls="-", lw=1.5)
  plot_manager.save_figure(fig, "test.png")



## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################

if __name__ == "__main__":
  main()


## END OF SCRIPT