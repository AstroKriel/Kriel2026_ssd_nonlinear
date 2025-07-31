import numpy as np
from pathlib import Path
from jormi.ww_io import io_manager, json_files


def extract_data_from_sim(samples: np.ndarray, model: str):
  init_energy    = 10 ** samples[:, 0]
  sat_energy     = 10 ** samples[:, 1]
  gamma_exp      = samples[:, 2]
  nl_start_time  = samples[:, 3]
  sat_start_time = samples[:, 4]
  if   model == "linear":    nl_exponent = np.ones_like(gamma_exp)
  elif model == "quadratic": nl_exponent = 2.0 * np.ones_like(gamma_exp)
  elif model == "free":      nl_exponent = samples[:, 5]
  else: raise ValueError(f"Unknown model type: {model}")
  nl_start_energy = init_energy * np.exp(gamma_exp * nl_start_time)
  nl_duration     = sat_start_time - nl_start_time
  gamma_nl        = (sat_energy - nl_start_energy) / (nl_duration ** nl_exponent)
  return {
    "gamma_exp"   : gamma_exp,
    "gamma_nl"    : gamma_nl,
    "nl_duration" : nl_duration,
    "sat_energy"  : sat_energy,
    "nl_exponent" : nl_exponent,
  }


class EnsembleAverager:
  model_types   = ["free", "linear", "quadratic"]
  quantity_keys = ["gamma_exp", "gamma_nl", "nl_duration", "sat_energy", "nl_exponent"]

  def __init__(self, directories):
    self.directories = directories
    self.sim_summary = {}

  def run(self):
    for model_type in self.model_types:
      combined_data = {
        quantity_key : []
        for quantity_key in self.quantity_keys
      }
      for directory in self.directories:
        data_path = io_manager.combine_file_path_parts([
          directory, model_type, f"stage2_{model_type}_fitted_posterior_samples.npy"
        ])
        if not io_manager.does_file_exist(data_path): continue
        sim_data = np.load(data_path)
        extracted_data = extract_data_from_sim(sim_data, model_type)
        for quantity_key in combined_data:
          combined_data[quantity_key].append(extracted_data[quantity_key])
      self.sim_summary[model_type] = {}
      for quantity_key, samples in combined_data.items():
        if not samples:
          self.sim_summary[model_type][quantity_key] = {"p16": None, "p50": None, "p84": None}
          continue
        flat_samples = np.concatenate(samples)
        self.sim_summary[model_type][quantity_key] = {
          "p16": float(np.percentile(flat_samples, 16)),
          "p50": float(np.percentile(flat_samples, 50)),
          "p84": float(np.percentile(flat_samples, 84)),
        }
    return self.sim_summary

def main():
  base_directory = Path("/scratch/jh2/nk7952/kriel2025_nl_data/").resolve()
  all_directories = io_manager.ItemFilter(
    include_string = ["Mach", "Re", "Pm", "Nres"]
  ).filter(
    directory = base_directory
  )
  sim_suites = set([
    str(directory).split("/")[-1].split("v")[0]
    for directory in all_directories
  ])
  all_results = {}
  for sim_suite in sorted(sim_suites):
    directories_in_suite = [
      directory
      for directory in all_directories
      if sim_suite in str(directory)
    ]
    sim_averager = EnsembleAverager(directories_in_suite)
    all_results[sim_suite] = sim_averager.run()
  json_files.save_dict_to_json_file("./mcmc_sim_summary.json", all_results, overwrite=True)

if __name__ == "__main__":
  main()