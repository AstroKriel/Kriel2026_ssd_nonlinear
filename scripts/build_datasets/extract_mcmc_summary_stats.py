import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.utils import list_utils

def extract_from_mcmc_data(samples: numpy.ndarray, model: str):
  init_energy    = 10 ** samples[:, 0]
  sat_energy     = 10 ** samples[:, 1]
  gamma_exp      = samples[:, 2]
  nl_start_time  = samples[:, 3]
  sat_start_time = samples[:, 4]
  if   model == "linear":    nl_exponent = numpy.ones_like(gamma_exp)
  elif model == "quadratic": nl_exponent = 2.0 * numpy.ones_like(gamma_exp)
  elif model == "free":      nl_exponent = samples[:, 5]
  else: raise ValueError(f"Unknown model type: {model}")
  nl_start_energy = init_energy * numpy.exp(gamma_exp * nl_start_time)
  nl_duration     = sat_start_time - nl_start_time
  gamma_nl        = (sat_energy - nl_start_energy) / (nl_duration ** nl_exponent)
  return {
    "gamma_exp"      : gamma_exp,
    "gamma_nl"       : gamma_nl,
    "nl_start_time"  : nl_start_time,
    "sat_start_time" : sat_start_time,
    "nl_duration"    : nl_duration,
    "sat_energy"     : sat_energy,
    "nl_exponent"    : nl_exponent,
  }

class EnsembleAverager:
  model_types = [
    "free",
    "linear",
    "quadratic"
  ]
  quantity_keys = [
    "gamma_exp",
    "gamma_nl",
    "nl_duration",
    "sat_energy",
    "nl_exponent"
  ]

  def __init__(self, sim_directories):
    self.sim_directories = sim_directories
    self.fit_summary = {}
    self.sim_params = None
    self.exracted_data = False

  def run(self):
    ## for each fit-model
    for model_type in self.model_types:
      ## initialise quantities we want to accumulate over the different simulation instances
      combined_data = {
        quantity_key : []
        for quantity_key in self.quantity_keys
      }
      ## loop over the different simulation instances
      for sim_directory in self.sim_directories:
        print("Looking at:", sim_directory)
        ## get simulation data
        sim_data_path = io_manager.combine_file_path_parts([ sim_directory, "dataset.json" ])
        sim_data = json_files.read_json_file_into_dict(sim_data_path)
        target_Mach = sim_data["plasma_params"]["target_Mach"]
        target_Re = sim_data["plasma_params"]["target_Re"]
        if target_Mach > 1:
          _model_type = f"{model_type}_better_binning"
        else: _model_type = model_type
        mcmc_data_path = io_manager.combine_file_path_parts([
          sim_directory, _model_type, f"stage2_{_model_type}_fitted_posterior_samples.npy"
        ])
        if not io_manager.does_file_exist(mcmc_data_path):
          print(f"Simulation does not have mcmc data fitted with `{_model_type}` for: {sim_directory}\n")
          continue
        mcmc_data = numpy.load(mcmc_data_path)
        extracted_data = extract_from_mcmc_data(mcmc_data, _model_type)
        for quantity_key in combined_data:
          combined_data[quantity_key].append(extracted_data[quantity_key])
        ## only extract sim_params once (arbitrarily, from the linear fit)
        if not self.exracted_data:
          nu = 0.5 * target_Mach / target_Re
          t_turb = sim_data["plasma_params"]["t_turb"]
          full_time_values = numpy.array(sim_data["measured_data"]["time_values"])
          full_Mach_energy = numpy.array(sim_data["measured_data"]["rms_Mach_values"])
          median_nl_start_time = numpy.median(extracted_data["nl_start_time"])
          start_index = list_utils.find_first_crossing(full_time_values / t_turb, 5)
          end_index = list_utils.find_first_crossing(full_time_values, median_nl_start_time)
          kinematic_Mach_values = full_Mach_energy[start_index : end_index]
          kinematic_Re_values = 0.5 * kinematic_Mach_values / nu
          self.sim_params = {
            "Mach" : {
              "p16": float(numpy.percentile(kinematic_Mach_values, 16)),
              "p50": float(numpy.percentile(kinematic_Mach_values, 50)),
              "p84": float(numpy.percentile(kinematic_Mach_values, 84))
            },
            "Re" : {
              "p16": float(numpy.percentile(kinematic_Re_values, 16)),
              "p50": float(numpy.percentile(kinematic_Re_values, 50)),
              "p84": float(numpy.percentile(kinematic_Re_values, 84))
            }
          }
          self.exracted_data = True
        print(" ")
      self.fit_summary[model_type] = {}
      for quantity_key, samples in combined_data.items():
        if not samples:
          self.fit_summary[model_type][quantity_key] = {"p16": None, "p50": None, "p84": None}
          continue
        flat_samples = numpy.concatenate(samples)
        self.fit_summary[model_type][quantity_key] = {
          "p16": float(numpy.percentile(flat_samples, 16)),
          "p50": float(numpy.percentile(flat_samples, 50)),
          "p84": float(numpy.percentile(flat_samples, 84))
        }
    return {
      "fit_summaries": self.fit_summary,
      "sim_params": self.sim_params
    }

def main():
  base_directory = Path("/scratch/jh2/nk7952/ssd_sims").resolve()
  all_directories = io_manager.ItemFilter(
    include_string = ["Mach", "Re", "Pm", "Nres"]
  ).filter(
    directory = base_directory
  )
  sim_suites = set([
    str(sim_directory).split("/")[-1].split("v")[0]
    for sim_directory in all_directories
  ])
  all_results = {}
  for sim_suite in sorted(sim_suites):
    directories_in_suite = [
      sim_directory
      for sim_directory in all_directories
      if sim_suite in str(sim_directory)
    ]
    ## average over the different simulation instances (at a particular resolution)
    sim_averager = EnsembleAverager(directories_in_suite)
    all_results[sim_suite] = sim_averager.run()
  json_files.save_dict_to_json_file("./summary_stats.json", all_results, overwrite=True)

if __name__ == "__main__":
  main()

## .