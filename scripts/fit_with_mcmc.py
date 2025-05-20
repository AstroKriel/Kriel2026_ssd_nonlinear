## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from jormi.utils import list_utils
from jormi.ww_io import io_manager
from my_utils import ww_sims, ww_mcmc


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################

def compute_median_params_from_kde(kde, num_samples=10000):
  samples = kde.resample(num_samples)
  return tuple(numpy.median(samples, axis=1))


## ###############################################################
## MCMC mcmc_routine
## ###############################################################
def mcmc_routine(sim_directory, level1_output_directory, verbose=True):
  sim_name = ww_sims.get_sim_name(sim_directory)
  level2_output_directory = io_manager.combine_file_path_parts([ level1_output_directory, sim_name ])
  io_manager.init_directory(level2_output_directory, verbose=False)
  data_dict = ww_sims.load_data(sim_directory, num_samples=500)
  x_values  = data_dict["time"]
  y_values  = data_dict["magnetic_energy"]
  ## stage 1 MCMC fitter
  stage1_initial_params = (
    -20, # log10(E_init)
    0.5, # log10(E_sat)
    0.5 * numpy.max(x_values) # gammma
  )
  stage1_mcmc = ww_mcmc.MCMCStage1Routine(
    output_directory = level2_output_directory,
    x_values         = x_values,
    y_values         = y_values,
    initial_params   = stage1_initial_params,
    verbose          = verbose,
    plot_kde         = True
  )
  stage1_mcmc.estimate_posterior()
  ## stage 2 MCMC fitter
  stage2_prior_kde = stage1_mcmc.output_posterior_kde
  stage1_median_output_params = compute_median_params_from_kde(stage2_prior_kde)
  stage1_median_transition_time = numpy.median(stage1_mcmc.fitted_posterior_samples[:,2])
  stage2_initial_params = (
    stage1_median_output_params[0], # log10(E_init)
    stage1_median_output_params[1], # log10(E_sat)
    stage1_median_output_params[2], # gammma
    0.5 * stage1_median_transition_time, # t_nl
    0.5 * (numpy.max(x_values) + stage1_median_transition_time) # t_sat
  )
  approx_transition_index = list_utils.get_index_of_closest_value(x_values, stage1_median_transition_time)
  stage2_likelihood_sigma = numpy.std(y_values[approx_transition_index:])
  stage2_mcmc = ww_mcmc.MCMCStage2Routine(
    output_directory = level2_output_directory,
    x_values         = x_values,
    y_values         = y_values,
    likelihood_sigma = stage2_likelihood_sigma,
    initial_params   = stage2_initial_params,
    prior_kde        = stage2_prior_kde,
    verbose          = verbose,
    plot_kde         = True
  )
  stage2_mcmc.estimate_posterior(num_walkers=50, num_steps=3000)
  ww_mcmc.plot_final_fits.PlotFinalFits(stage2_mcmc).plot()


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  script_directory = io_manager.get_caller_directory()
  output_directory = io_manager.combine_file_path_parts([ script_directory, "mcmc_fits" ])
  io_manager.init_directory(output_directory, verbose=False)
  mcmc_routine(
    sim_directory           = "/scratch/jh2/nk7952/Re1500/Mach2/Pm1/576v4",
    level1_output_directory = output_directory,
  )


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT