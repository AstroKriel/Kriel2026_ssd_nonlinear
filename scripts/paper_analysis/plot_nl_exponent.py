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
  fig, ax = plot_manager.create_figure()
  cmap_nl_exponent, norm_nl_exponent = add_color.create_cmap(
    cmap_name = "bwr",
    vmin      = 1.0,
    vmax      = 2.0,
  )
  data_directory = "/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_ssd_nl/datasets/"
  summary_path = Path(f"{data_directory}/summary_stats.json")
  all_results = json_files.read_json_file_into_dict(summary_path)
  for sim_suite, sim_data in all_results.items():
    print("Looking at:", sim_suite)
    if sim_data["sim_params"]["Mach"]["p50"] < 1:
      extra_tag = ""
    else: extra_tag = "_better_binning"
    ## dataset
    nl_exponent_p50 = sim_data["fit_summaries"][f"free{extra_tag}"]["nl_exponent"]["p50"]
    if nl_exponent_p50 is None:
      continue
    nl_exponent_color = cmap_nl_exponent(norm_nl_exponent(nl_exponent_p50))
    ## mach number
    Mach_stats = sim_data["sim_params"]["Mach"]
    log10_Mach_p50 = numpy.log10(Mach_stats["p50"])
    log10_Mach_err_lower = log10_Mach_p50 - numpy.log10(Mach_stats["p16"])
    log10_Mach_err_upper = numpy.log10(Mach_stats["p84"]) - log10_Mach_p50
    ## reynolds number
    Re_stats = sim_data["sim_params"]["Re"]
    log10_Re_p50 = numpy.log10(Re_stats["p50"])
    log10_Re_err_lower = log10_Re_p50 - numpy.log10(Re_stats["p16"])
    log10_Re_err_upper = numpy.log10(Re_stats["p84"]) - log10_Re_p50
    if log10_Re_p50 < 3:
      continue
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
      log10_Mach_p50 + 0.05 * random.uniform(-1, 1),
      log10_Re_p50 + 0.05 * random.uniform(-1, 1),
      xerr = [
        [log10_Mach_err_lower],
        [log10_Mach_err_upper],
      ],
      yerr = [
        [log10_Re_err_lower],
        [log10_Re_err_upper],
      ],
      fmt=marker, color=nl_exponent_color, mec="black", markersize=10, capsize=3, zorder=zorder
    )
  ax.set_xlim([-1.5, 1.0])
  ax.set_ylim([3.0, 3.8])
  ax.axvline(x=0, ls="--", color="black", lw=1.5, zorder=5)
  ax.set_xlabel(r"$\log_{10}(\mathcal{M})$")
  ax.set_ylabel(r"$\log_{10}(\mathrm{Re})$")
  add_color.add_cbar_from_cmap(
    ax    = ax,
    cmap  = cmap_nl_exponent,
    norm  = norm_nl_exponent,
    label = r"$p$",
    side  = "top",
  )
  add_annotations.add_custom_legend(
    ax          = ax,
    artists     = [ "o", "s", "D" ],
    labels      = [ r"$288^3$", r"$576^3$", r"$1152^3$" ],
    colors      = [ "k", "k", "k" ],
    marker_size = 8,
    line_width  = 1.5,
    fontsize    = 16,
    text_color  = "k",
    position    = "upper left",
    anchor      = (0.0, 0.85),
  )
  plot_manager.save_figure(fig, f"nl_exponent_scaling.png")


if __name__ == "__main__":
  main()


## end