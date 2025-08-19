## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
import argparse
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from my_mcmc_fitting_routine.mcmc_stage_1 import Stage1MCMCRoutine
from my_mcmc_fitting_routine.mcmc_stage_2 import Stage2MCMCRoutine_free
from my_mcmc_fitting_routine.mcmc_stage_2 import Stage2MCMCRoutine_linear
from my_mcmc_fitting_routine.mcmc_stage_2 import Stage2MCMCRoutine_quadratic
from my_mcmc_fitting_routine.plot_final_fits import PlotFinalFits


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################

def compute_median_params_from_kde(kde, num_samples=10000):
  samples = kde.resample(num_samples)
  return tuple(numpy.median(samples, axis=1))

def compute_binned_data(x_values, y_values, num_bins):
  x_bin_edges = numpy.linspace(0, numpy.max(x_values), num_bins+1)
  x_bin_centers = 0.5 * (x_bin_edges[1:] + x_bin_edges[:-1])
  x_bin_indices = numpy.digitize(x_values, x_bin_edges) - 1
  y_ave_s = numpy.zeros(num_bins)
  y_std_s = numpy.zeros(num_bins)
  log10_y_ave_s = numpy.zeros(num_bins)
  log10_y_std_s = numpy.zeros(num_bins)
  for bin_index in range(num_bins):
    bin_mask = (x_bin_indices == bin_index)
    if numpy.any(bin_mask):
      y_values_in_bin = numpy.array(y_values)[bin_mask]
      log10_y_values_in_bin = numpy.log10(y_values_in_bin)
      y_ave_s[bin_index] = numpy.mean(y_values_in_bin)
      y_std_s[bin_index] = numpy.std(y_values_in_bin)
      log10_y_ave_s[bin_index] = numpy.mean(log10_y_values_in_bin)
      log10_y_std_s[bin_index] = numpy.std(log10_y_values_in_bin)
    else:
      y_ave_s[bin_index] = numpy.nan
      y_std_s[bin_index] = numpy.nan
      log10_y_ave_s[bin_index] = numpy.nan
      log10_y_std_s[bin_index] = numpy.nan
  return {
    "x_bin_centers": x_bin_centers,
    "y_ave_s": y_ave_s,
    "y_std_s": y_std_s,
    "log10_y_ave_s": log10_y_ave_s,
    "log10_y_std_s": log10_y_std_s
  }


## ###############################################################
## PROGRAM MAIN
## ###############################################################

def main():
  ## collect user arguments
  parser = argparse.ArgumentParser(description="Run MCMC fitting routine.")
  parser.add_argument("-data_directory", type=str, required=True)
  parser.add_argument("-model", type=str, required=True, choices=["linear", "quadratic", "free"])
  parser.add_argument("-num_bins", type=int, default=None, help="Number of bins. Default: one bin per eddy-turnover time.")
  args = parser.parse_args()
  data_directory = Path(args.data_directory).resolve()
  model_name = args.model
  num_bins = args.num_bins
  if num_bins is None:
    binning_notice = "one bin per eddy-turnover time"
    binning_tag = "bin_per_t0"
  else:
    binning_notice = f"{num_bins} bins"
    binning_tag = f"{num_bins}bins"
  print(f"Looking at: {data_directory}")
  print(f"Fitting the {model_name}-model to the nonlinear (backreaction) phase with {binning_notice}.")
  ## read in magnetic energy evolution
  output_directory = io_manager.combine_file_path_parts([ data_directory, model_name, binning_tag ])
  io_manager.init_directory(output_directory)
  data_filepath = io_manager.combine_file_path_parts([ data_directory, "dataset.json" ])
  data_dict = json_files.read_json_file_into_dict(data_filepath)
  ## subset the simulation domain: roughly half of the data points should make up the growth phase
  full_time_values = numpy.array(data_dict["measured_data"]["time_values"])
  full_magnetic_energy = numpy.array(data_dict["measured_data"]["magnetic_energy_values"])
  t_turb = data_dict["plasma_params"]["t_turb"]
  max_total_time = numpy.max(full_time_values)
  max_subset_time = max_total_time # initialise
  max_sat_fraction_of_subset_time = 0.35
  ## loop until the transition time is late enough, or we can't trim more
  while True:
    ## subset data
    time_mask = (full_time_values <= max_subset_time)
    subset_time_values = full_time_values[time_mask]
    subset_magnetic_energy = full_magnetic_energy[time_mask]
    max_subset_time = numpy.max(subset_time_values)
    binned_data = compute_binned_data(
      x_values = subset_time_values,
      y_values = subset_magnetic_energy,
      num_bins = int(numpy.max(subset_time_values) / t_turb) if (num_bins is None) else num_bins
    )
    stage1_initial_params = (
      -20, # log10(E_init)
      0.5, # log10(E_sat)
      0.5 * max_subset_time # transition time
    )
    print("Running stage 1.")
    stage1_mcmc = Stage1MCMCRoutine(
      output_directory        = output_directory,
      time_values             = binned_data["x_bin_centers"],
      ave_log10_energy_values = binned_data["log10_y_ave_s"],
      std_log10_energy_values = binned_data["log10_y_std_s"],
      initial_params          = stage1_initial_params,
      plot_posterior_kde      = False
    )
    stage1_mcmc.estimate_posterior()
    stage1_median_transition_time = numpy.median(stage1_mcmc.fitted_posterior_samples[:, 2])
    sat_fraction_of_subset_time = stage1_median_transition_time / max_subset_time
    sat_percent_of_subset_time = 100 * sat_fraction_of_subset_time
    print(f"Estimated stage 1 transition time: {stage1_median_transition_time:.2f} ({sat_percent_of_subset_time:.1f}% of max trimmed time)")
    if sat_fraction_of_subset_time >= max_sat_fraction_of_subset_time: break
    max_subset_time *= 0.85 # trim off 15% of tail
    print(f"Trimmed to {max_subset_time:.2f}, re-running stage 1...")
  stage1_mcmc.plot_posterior_kde = True
  stage1_mcmc._make_plots()
  ## extract key outputs from stage 1
  stage2_prior_kde = stage1_mcmc.output_posterior_kde
  ## build initial guess for stage 2
  stage1_median_output_params = compute_median_params_from_kde(stage2_prior_kde)
  stage2_initial_params = (
    stage1_median_output_params[0], # log10(E_init)
    stage1_median_output_params[1], # log10(E_sat)
    stage1_median_output_params[2], # gamma_exp
    0.5 * stage1_median_transition_time, # t_nl
  )
  ## run stage 2 fitter
  print("Running stage 2.")
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
      initial_params     = stage2_initial_params,
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