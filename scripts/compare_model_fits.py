import numpy
import random
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color


def extract_key_param_samples(fitted_posterior_samples):
  beta_samples = fitted_posterior_samples[:,5]
  return beta_samples

def main():
  base_directory = Path("/scratch/jh2/nk7952/kriel2025_nl_data/").resolve()
  directories = io_manager.ItemFilter(
    include_string = ["Mach", "Re", "Pm", "Nres576"]
  ).filter(
    directory = base_directory
  )
  for directory in directories:
    sim_path = io_manager.combine_file_path_parts([ directory, "dataset.json" ])
    sim_dict = json_files.read_json_file_into_dict(sim_path, verbose=False)
    Mach_number = sim_dict["plasma_params"]["Mach"]
    Re_number = sim_dict["plasma_params"]["Re"]
    if Re_number < 1000: continue
    if Mach_number < 1: continue
    aics = []
    for model_name in ["linear", "quadratic"]:
      data_path = io_manager.combine_file_path_parts([ directory, model_name, f"stage2_{model_name}_fitted_log_likelihoods.npy" ])
      if not io_manager.does_file_exist(data_path): continue
      log_likelihoods = numpy.load(data_path)
      max_ll = numpy.max(log_likelihoods)
      aic = 10 - 2 * max_ll
      aics.append(aic)
    if len(aics) < 2:
      print("Look at:", directory)
      continue
    min_aic = numpy.min(aics)
    delta0 = numpy.exp(-(aics[0] - min_aic))
    delta1 = numpy.exp(-(aics[1] - min_aic))
    p0 = delta0 / (delta0 + delta1)
    p1 = delta1 / (delta0 + delta1)
    if   (Mach_number > 1) and (p0 < p1): result = "!"
    elif (Mach_number < 1) and (p0 > p1): result = "!"
    else: result = "x"
    print(result, Mach_number, p0, p1)


if __name__ == "__main__":
  main()


## end