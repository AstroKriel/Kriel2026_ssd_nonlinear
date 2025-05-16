## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from pathlib import Path
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager
from jormi.parallelism import independent_tasks
from my_utils import ww_sims, ww_mcmc


# ## ###############################################################
# ## FINAL ENERGY MODEL
# ## ###############################################################
# def energy_model(time, stage1_params, stage2_params):
#   (log10_init_energy, _, gamma) = stage1_params
#   (start_nl_time, start_sat_time, log10_sat_energy) = stage2_params
#   ## mask different ssd phases
#   mask_exp_phase = time < start_nl_time
#   mask_nl_phase  = (start_nl_time <= time) & (time < start_sat_time)
#   mask_sat_phase = start_sat_time < time
#   ## calculate model constants
#   init_energy     = 10**log10_init_energy
#   start_nl_energy = init_energy * numpy.exp(gamma * start_nl_time)
#   sat_energy      = 10**log10_sat_energy
#   alpha           = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
#   ## model energy evolution
#   energy = numpy.zeros_like(time)
#   energy[mask_exp_phase] = init_energy * numpy.exp(gamma * time[mask_exp_phase])
#   energy[mask_nl_phase]  = start_nl_energy + alpha * (time[mask_nl_phase] - start_nl_time)
#   energy[mask_sat_phase] = sat_energy
#   return energy


## ###############################################################
## MCMC ROUTINE
## ###############################################################
def routine(sim_directory, level1_output_directory, verbose=True):
  sim_name = ww_sims.get_sim_name(sim_directory)
  fig_file_name  = f"{sim_name}_fit.png"
  json_file_name = f"{sim_name}_params.json"
  fig_file_path  = io_manager.combine_file_path_parts([ level1_output_directory, fig_file_name ])
  json_file_path = io_manager.combine_file_path_parts([ level1_output_directory, json_file_name ])
  level2_output_directory = io_manager.combine_file_path_parts([ level1_output_directory, sim_name ])
  io_manager.init_directory(level2_output_directory, verbose=False)
  ## load and interpolate data
  data_dict = ww_sims.load_data(sim_directory, num_samples=70)
  ## stage 1 MCMC fitter
  stage1_mcmc = ww_mcmc.MCMCStage1Routine(
    output_directory = level2_output_directory,
    x_values         = data_dict["time"],
    y_values         = data_dict["magnetic_energy"],
    verbose          = verbose
  )
  stage1_mcmc.sample_posterior()
  # ## stage 2 MCMC fitter
  # stage2_mcmc = ww_mcmc.MCMCStage2Routine(
  #   output_directory = level2_output_directory,
  #   x_values         = time,
  #   y_values         = measured_energy,
  #   stage1_params    = stage1_params,
  #   verbose          = verbose
  # )
  # stage2_mcmc.sample_posterior()
  # ## plot final fit
  # fig, axs = plot_manager.create_figure(num_rows=2, share_x=True)
  # fig_data_params = dict(color="blue", marker="o", ms=3, lw=1)
  # fig_fit_params  = dict(color="red", lw=2)
  # modelled_energy = energy_model(time, stage1_params, stage2_params)
  # axs[0].plot(data_dict["time"], data_dict["magnetic_energy"], **fig_data_params)
  # axs[1].plot(data_dict["time"], numpy.log10(data_dict["magnetic_energy"]), **fig_data_params)
  # axs[0].plot(data_dict["time"], modelled_energy, **fig_fit_params)
  # axs[1].plot(data_dict["time"], numpy.log10(modelled_energy), **fig_fit_params)
  # axs[0].set_ylabel(r"$\mathrm{energy}$")
  # axs[1].set_ylabel(r"$\log_{10}(\mathrm{energy})$")
  # axs[1].set_xlabel(r"time")
  # plot_manager.save_figure(fig, fig_file_path, verbose=verbose)
  # fit_params_dict = {"stage1_params": stage1_params, "stage2_params": stage2_params}
  # json_files.save_dict_to_json_file(
  #   file_path  = json_file_path,
  #   input_dict = fit_params_dict,
  #   overwrite  = True,
  #   verbose    = verbose
  # )


## ###############################################################
## FIT ALL SSD SIMULATIONS
## ###############################################################
def fit_all_sims_in_parallel(output_directory):
  file_names_in_output_directory = [
    file.name
    for file in output_directory.iterdir()
    if file.is_file()
  ]
  sim_directories = sorted(Path("/scratch").glob("*/nk7952/Re*/Mach*/Pm1/576*"))
  sim_directories = [
    sim_directory
    for sim_directory in sim_directories
    if f"{ww_sims.get_sim_name(sim_directory)}_params.json" in file_names_in_output_directory # recompute all
  ]
  print("Will be looking at all of the following:")
  [
    print(str(sim_directory))
    for sim_directory in sim_directories
  ]
  print(" ")
  args_list = [
    (sim_directory, output_directory, False)
    for sim_directory in sim_directories
  ]
  independent_tasks.run_in_parallel(
    func            = routine,
    args_list       = args_list,
    num_procs       = 4,
    timeout_seconds = 5 * 60,
    show_progress   = True
  )


## ###############################################################
## ONLY FIT A SINGLE SIMULATION
## ###############################################################
def fit_single_sim(output_directory):
  sim_directory = "/scratch/jh2/nk7952/Re1500/Mach2/Pm1/576v4"
  routine(sim_directory, output_directory)


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  script_directory = io_manager.get_caller_directory()
  output_directory = io_manager.combine_file_path_parts([ script_directory, "mcmc_fits" ])
  io_manager.init_directory(output_directory, verbose=False)
  # fit_all_sims_in_parallel(output_directory)
  fit_single_sim(output_directory)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT