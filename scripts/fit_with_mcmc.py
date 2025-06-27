## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
import argparse
from pathlib import Path
from jormi.utils import list_utils
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
  ## collect user arguments
  parser = argparse.ArgumentParser(description="Run MCMC fitting routine.")
  parser.add_argument("-data_directory", type=str, required=True)
  parser.add_argument("-model", type=str, required=True, choices=["linear", "quadratic", "free"])
  args           = parser.parse_args()
  data_directory = Path(args.data_directory).resolve()
  model_name     = args.model
  print(f"Looking at: {data_directory}")
  print(f"Fitting the {model_name}-model to the nonlinear (backreaction) phase")
  ## read in magnetic energy evolution
  output_directory = io_manager.combine_file_path_parts([ data_directory, model_name ])
  io_manager.init_directory(output_directory)
  data_path   = io_manager.combine_file_path_parts([ data_directory, "dataset.json" ])
  data_dict   = json_files.read_json_file_into_dict(data_path)
  time_values = data_dict["data"]["time"]
  t_turb      = data_dict["plasma_params"]["t_turb"]
  binned_data = mcmc_utils.compute_binned_data(
    x_values = time_values,
    y_values = data_dict["data"]["magnetic_energy"],
    num_bins = int(numpy.floor(numpy.max(time_values) / t_turb)) # bin per uncorrelated eddy
  )
  ## build initial guess for stage 1: exponential + saturation
  stage1_initial_params = (
    -20, # log10(E_init)
    0.5, # log10(E_sat)
    0.5 * numpy.max(time_values) # gammma
  )
  ## run stage 1 fitter
  stage1_mcmc = Stage1MCMCRoutine(
    output_directory        = output_directory,
    time_values             = binned_data["x_bin_centers"],
    ave_log10_energy_values = binned_data["log10_y_ave_s"],
    std_log10_energy_values = binned_data["log10_y_std_s"],
    initial_params          = stage1_initial_params,
    plot_posterior_kde      = True
  )
  stage1_mcmc.estimate_posterior()
  ## extract key outputs from stage 1
  stage1_median_transition_time = numpy.median(stage1_mcmc.fitted_posterior_samples[:,2])
  stage2_prior_kde = stage1_mcmc.output_posterior_kde
  ## build initial guess for stage 2: exponential + linear backreaction + saturation
  stage1_median_output_params = mcmc_utils.compute_median_params_from_kde(stage2_prior_kde)
  stage2_initial_params = (
    stage1_median_output_params[0], # log10(E_init)
    stage1_median_output_params[1], # log10(E_sat)
    stage1_median_output_params[2], # gammma
    0.5 * stage1_median_transition_time, # t_nl
    0.5 * (numpy.max(time_values) + stage1_median_transition_time) # t_sat
  )
  ## run stage 2 fitter
  if model_name == "linear":
    stage2_mcmc = Stage2MCMCRoutine_linear(
      output_directory   = output_directory,
      time_values        = binned_data["x_bin_centers"],
      ave_energy_values  = binned_data["y_ave_s"],
      std_energy_values  = binned_data["y_std_s"],
      initial_params     = stage2_initial_params,
      prior_kde          = stage2_prior_kde,
      plot_posterior_kde = True
    )
  elif model_name == "quadratic":
    stage2_mcmc = Stage2MCMCRoutine_quadratic(
      output_directory   = output_directory,
      time_values        = binned_data["x_bin_centers"],
      ave_energy_values  = binned_data["y_ave_s"],
      std_energy_values  = binned_data["y_std_s"],
      initial_params     = stage2_initial_params,
      prior_kde          = stage2_prior_kde,
      plot_posterior_kde = True
    )
  else:
    stage2_mcmc = Stage2MCMCRoutine_free(
      output_directory   = output_directory,
      time_values        = binned_data["x_bin_centers"],
      ave_energy_values  = binned_data["y_ave_s"],
      std_energy_values  = binned_data["y_std_s"],
      initial_params     = stage2_initial_params + (1.5,), # tuples are immutable
      prior_kde          = stage2_prior_kde,
      plot_posterior_kde = True
    )
  stage2_mcmc.estimate_posterior()
  ## plot the measured vs modelled energy evolution (both linear and log10-transformed energy)
  PlotFinalFits(stage2_mcmc).plot()


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################

if __name__ == "__main__":
  main()


## END OF SCRIPT