import numpy
from pathlib import Path
from jormi.ww_io import json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color

mcmc_model = "linear"
extra_tag = ""
# extra_tag = "_better_binning"
fit_key = f"{mcmc_model}{extra_tag}"

x_min, x_max = 3.1, 3.75
y_min, y_max = -0.45, 0.45

def main():
  summary_path = Path("/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_ssd_nl/datasets/summary_stats.json")
  all_results = json_files.read_json_file_into_dict(summary_path)

  fig, ax = plot_manager.create_figure()
  cmap_Mach, norm_Mach = add_color.create_cmap(
    cmap_name = "cmr.holly_r",
    vmin = -1.5,
    vmid = 0,
    vmax = 0.5,
    cmin = 0.15,
    cmax = 0.85,
  )

  for sim_suite, sim_data in all_results.items():
    if fit_key not in sim_data["fit_summaries"]:
      continue
    gamma_exp_stats = sim_data["fit_summaries"][fit_key]["gamma_exp"]
    if gamma_exp_stats["p50"] is None:
      continue

    Mach_stats = sim_data["sim_params"]["Mach"]
    Mach_p50 = Mach_stats["p50"]
    if Mach_p50 <= 0:
      continue
    log10_Mach = numpy.log10(Mach_p50)

    Re_stats = sim_data["sim_params"]["Re"]
    Re_p50 = Re_stats["p50"]
    if Re_p50 < 1000:
      continue
    log10_Re_p50 = numpy.log10(Re_p50)
    log10_Re_err_lower = log10_Re_p50 - numpy.log10(Re_stats["p16"])
    log10_Re_err_upper = numpy.log10(Re_stats["p84"]) - log10_Re_p50

    scaled_gamma_exp_p50 = numpy.array(gamma_exp_stats["p50"]) / Mach_p50
    scaled_gamma_exp_normed_p50 = numpy.log10(scaled_gamma_exp_p50)
    scaled_gamma_exp_normed_err_lower = (
      numpy.log10(scaled_gamma_exp_p50) - numpy.log10(numpy.array(gamma_exp_stats["p16"]) / Mach_p50)
    )
    scaled_gamma_exp_normed_err_upper = (
      numpy.log10(numpy.array(gamma_exp_stats["p84"]) / Mach_p50) - numpy.log10(scaled_gamma_exp_p50)
    )

    Mach_color = cmap_Mach(norm_Mach(log10_Mach))

    if "288" in sim_suite:
      marker, zorder = "o", 1
    elif "576" in sim_suite:
      marker, zorder = "s", 3
    elif "1152" in sim_suite:
      marker, zorder = "D", 5
    else:
      print("Could not determine resolution for:", sim_suite)
      continue

    ax.errorbar(
      log10_Re_p50 ,
      scaled_gamma_exp_normed_p50,
      xerr=[[log10_Re_err_lower], [log10_Re_err_upper]],
      yerr=[[scaled_gamma_exp_normed_err_lower], [scaled_gamma_exp_normed_err_upper]],
      fmt=marker, markerfacecolor=Mach_color, zorder=zorder,
      markeredgecolor="black", ecolor="black", markersize=10, lw=2, capsize=3
    )

  ax.set_xlim([ x_min, x_max ])
  ax.set_ylim([ y_min, y_max ])
  x_values = numpy.linspace(3, 4, 100)

  plot_data.plot_wo_scaling_axis(
    ax=ax,
    x_values=x_values,
    y_values=numpy.log10(3e-2) + 0.5 * x_values,
    ls="--",
    lw=1.5,
    zorder=-15
  )
  plot_data.plot_wo_scaling_axis(
    ax=ax,
    x_values=x_values,
    y_values=numpy.log10(4.5e-2) + (1/3) * x_values,
    ls=":",
    lw=1.5,
    zorder=-15
  )

  rotation_bounds = (x_min, x_max, y_min, y_max)
  rotation1 = fit_data.get_line_angle_in_box(
    slope = 1/2,
    domain_bounds = rotation_bounds,
    domain_aspect_ratio = 6/4,
  )
  rotation2 = fit_data.get_line_angle_in_box(
    slope = 1/3,
    domain_bounds = rotation_bounds,
    domain_aspect_ratio = 6/4,
  )

  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.335,
    y_pos       = 0.725,
    label       = r"$3 \times 10^{-2}\, \mathrm{Re}^{1/2}$",
    x_alignment = "center",
    y_alignment = "center",
    rotate_deg  = rotation1,
    font_color  = cmap_Mach(norm_Mach(-1.5))
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.65,
    y_pos       = 0.235,
    label       = r"$4.5 \times 10^{-2}\, \mathrm{Re}^{1/3}$",
    x_alignment = "center",
    y_alignment = "center",
    rotate_deg  = rotation2,
    font_color  = cmap_Mach(norm_Mach(0.5))
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.95,
    y_pos       = 0.625,
    label       = r"growth regulated by the",
    x_alignment = "right",
    y_alignment = "top",
    font_color  = "black"
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.95,
    y_pos       = 0.525,
    label       = r"smallest kinetic fluctuations",
    x_alignment = "right",
    y_alignment = "top",
    font_color  = "black"
  )
  ax.set_xlabel(r"$\log_{10}(\mathrm{Re})$")
  ax.set_ylabel(r"$\log_{10}(\gamma_\mathrm{exp} \,t_0^{-1})$")
  add_color.add_cbar_from_cmap(
    ax       = ax,
    cmap     = cmap_Mach,
    norm     = norm_Mach,
    label    = r"$\log_{10}(\mathcal{M})$",
    side     = "top",
    fontsize = 22
  )
  add_annotations.add_custom_legend(
    ax           = ax,
    artists      = ["o", "s", "D"],
    labels       = [ r"$288^3$", r"$576^3$", r"$1152^3$" ],
    colors       = ["k"] * 3,
    marker_size  = 8,
    line_width   = 1.5,
    fontsize     = 16,
    text_color   = "k",
    position     = "upper left",
    anchor       = (-0.035, 1.0),
    num_cols     = 3,
    text_padding = 0.0
  )
  plot_manager.save_figure(fig, f"gamma_exp_scaling.png")


if __name__ == "__main__":
  main()
