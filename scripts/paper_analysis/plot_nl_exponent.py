import numpy
import random
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color


def extract_key_param_samples(fitted_posterior_samples):
  nl_exponent_samples = fitted_posterior_samples[:,5]
  return nl_exponent_samples

def main():
  base_directory = Path("/scratch/jh2/nk7952/kriel2025_nl_data/").resolve()
  directories = io_manager.ItemFilter(
    include_string = ["Mach", "Re", "Pm", "Nres"]
  ).filter(
    directory = base_directory
  )
  sim_suites = set([
    str(directory).split("/")[-1].split("v")[0]
    for directory in directories
  ])
  fig, ax = plot_manager.create_figure()
  cmap_beta, norm_beta = add_color.create_cmap(
    cmap_name = "bwr",
    vmin      = 1.0,
    vmax      = 2.0,
  )
  for sim_suite in sim_suites:
    print("Looking at:", sim_suite)
    directories_in_suite = [
      directory
      for directory in directories
      if sim_suite in str(directory)
    ]
    all_nl_exponent_samples = []
    for directory in directories_in_suite:
      sim_data_path = io_manager.combine_file_path_parts([ directory, "dataset.json" ])
      sim_data_dict = json_files.read_json_file_into_dict(sim_data_path, verbose=False)
      fit_data_path = io_manager.combine_file_path_parts([ directory, "free_better_binning", f"stage2_free_fitted_posterior_samples.npy" ])
      if not io_manager.does_file_exist(fit_data_path): continue
      fitted_posterior_samples = numpy.load(fit_data_path)
      nl_exponent_samples = extract_key_param_samples(fitted_posterior_samples)
      all_nl_exponent_samples.append(nl_exponent_samples)
      p50 = numpy.percentile(nl_exponent_samples, 50)
      if (sim_data_dict["plasma_params"]["target_Mach"] < 1) and (p50 > 1.5):
        print(f"not linear ({p50:.2f}): {directory}/free_better_binning/stage2_free_fitted_posteriors.png")
      elif (sim_data_dict["plasma_params"]["target_Mach"] > 1) and (p50 < 1.5):
        print(f"not quadratic ({p50:.2f}): {directory}/free_better_binning/stage2_free_fitted_posteriors.png")
    if len(all_nl_exponent_samples) == 0:
      print("Need to look at:", directory)
      continue
    nl_exponent_p50 = numpy.percentile(all_nl_exponent_samples, 50)
    nl_exponent_color = cmap_beta(norm_beta(nl_exponent_p50))
    sim_name = sim_data_dict["sim_name"]
    Mach_values = sim_data_dict["measured_data"]["rms_Mach_values"]
    Mach_p16, Mach_p50, Mach_p84 = numpy.percentile(numpy.log10(Mach_values), [16, 50, 84])
    Mach_err_lower = Mach_p50 - Mach_p16
    Mach_err_upper = Mach_p84 - Mach_p50
    Mach_number = sim_data_dict["plasma_params"]["target_Mach"]
    Re_number = sim_data_dict["plasma_params"]["target_Re"]
    Re_values = float(Re_number) / float(Mach_number) * numpy.array(Mach_values)
    Re_p16, Re_p50, Re_p84 = numpy.percentile(numpy.log10(Re_values), [16, 50, 84])
    Re_err_lower = Re_p50 - Re_p16
    Re_err_upper = Re_p84 - Re_p50
    if Re_number < 1000: continue
    if "288" in sim_suite:
      marker = "o"
      zorder = 1
    elif "576" in sim_suite:
      marker = "s"
      zorder = 3
    elif "1152" in sim_suite:
      marker = "D"
      zorder = 5
    else: print("error")
    ax.errorbar(
      Mach_p50 + 0.1 * random.uniform(-1, 1),
      Re_p50 + 0.1 * random.uniform(-1, 1),
      xerr = [
        [Mach_err_lower],
        [Mach_err_upper],
      ],
      yerr = [
        [Re_err_lower],
        [Re_err_upper],
      ],
      fmt=marker, color=nl_exponent_color, mec="black", markersize=10, capsize=3, zorder=zorder
    )
  ax.axvline(x=0, ls="--", color="black", lw=1.5, zorder=5)
  ax.set_xlabel(r"$\log_{10}(\mathcal{M})$")
  ax.set_ylabel(r"$\log_{10}(\mathrm{Re})$")
  add_color.add_cbar_from_cmap(
    ax    = ax,
    cmap  = cmap_beta,
    norm  = norm_beta,
    label = r"$p$",
    side  = "top",
  )
  add_annotations.add_custom_legend(
    ax             = ax,
    artists        = [ "o", "s", "D" ],
    labels         = [ r"$288^3$", r"$576^3$", r"$1152^3$" ],
    colors         = [ "k", "k", "k" ],
    marker_size    = 8,
    line_width     = 1.5,
    fontsize       = 16,
    text_color     = "k",
    position       = "upper left",
    anchor         = (0.0, 1.0),
    # enable_frame   = bool = False,
    # frame_alpha    = float = 0.5,
    # num_cols       = float = 1,
    # text_padding   = float = 0.5,
    # label_spacing  = float = 0.5,
    # column_spacing = float = 0.5,
  )
  plot_manager.save_figure(fig, f"nl_exponent_scaling_better_binning.png")


if __name__ == "__main__":
  main()


## end