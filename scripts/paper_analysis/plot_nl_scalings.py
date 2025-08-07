import numpy
from pathlib import Path
from jormi.ww_io import json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color


x_min, x_max = -1.5, 1.0
y0_min, y0_max = -6.5, 0.0
y1_min, y1_max = 0.0, 2.5

def main():
  summary_path = Path("/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_ssd_nl/datasets/summary_stats.json")
  all_results = json_files.read_json_file_into_dict(summary_path)

  fig, axs = plot_manager.create_figure(num_rows=2)
  cmap_Re, norm_Re = add_color.create_cmap(
    cmap_name = "Blues",
    vmin      = numpy.log10(100),
    vmax      = numpy.log10(5000),
  )

  for sim_suite, sim_data in all_results.items():
    gamma_nl_stats = sim_data["fit_summaries"]["linear"]["gamma_nl"]
    if gamma_nl_stats["p50"] is None:
      continue

    log10_gamma_nl_p50 = numpy.log10(gamma_nl_stats["p50"])
    log10_gamma_nl_err_lower = log10_gamma_nl_p50 - numpy.log10(gamma_nl_stats["p16"])
    log10_gamma_nl_err_upper = numpy.log10(gamma_nl_stats["p84"]) - log10_gamma_nl_p50

    duration_stats = sim_data["fit_summaries"]["linear"]["nl_duration"]
    if duration_stats["p50"] is None:
      continue

    log10_delta_t_p50 = numpy.log10(duration_stats["p50"])
    log10_delta_t_err_lower = log10_delta_t_p50 - numpy.log10(duration_stats["p16"])
    log10_delta_t_err_upper = numpy.log10(duration_stats["p84"]) - log10_delta_t_p50

    Mach_stats = sim_data["sim_params"]["Mach"]
    log10_Mach_p50 = numpy.log10(Mach_stats["p50"])
    log10_Mach_err_lower = log10_Mach_p50 - numpy.log10(Mach_stats["p16"])
    log10_Mach_err_upper = numpy.log10(Mach_stats["p84"]) - log10_Mach_p50

    Re_stats = sim_data["sim_params"]["Re"]
    Re_p50 = Re_stats["p50"]
    if Re_p50 < 1000:
      continue
    Re_color = cmap_Re(norm_Re(numpy.log10(Re_p50)))

    if "288" in sim_suite:
      marker, zorder = "o", 1
    elif "576" in sim_suite:
      marker, zorder = "s", 3
    elif "1152" in sim_suite:
      marker, zorder = "D", 5
    else:
      print("Could not determine resolution for:", sim_suite)
      continue

    axs[0].errorbar(
      log10_Mach_p50,
      log10_gamma_nl_p50,
      xerr=[[log10_Mach_err_lower], [log10_Mach_err_upper]],
      yerr=[[log10_gamma_nl_err_lower], [log10_gamma_nl_err_upper]],
      fmt=marker, markerfacecolor=Re_color, zorder=zorder,
      markeredgecolor="black", ecolor="black", markersize=10, lw=2, capsize=3
    )
    axs[1].errorbar(
      log10_Mach_p50,
      log10_delta_t_p50,
      xerr=[[log10_Mach_err_lower], [log10_Mach_err_upper]],
      yerr=[[log10_delta_t_err_lower], [log10_delta_t_err_upper]],
      fmt=marker, markerfacecolor=Re_color, zorder=zorder,
      markeredgecolor="black", ecolor="black", markersize=10, lw=2, capsize=3
    )


  axs[0].set_xticklabels([])
  axs[1].set_xlabel(r"$\log_{10}(\mathcal{M})$")
  axs[0].set_ylabel(r"$\log_{10}(\gamma_{\rm nl})$")
  axs[1].set_ylabel(r"$\log_{10}\big((t_{\rm sat} - t_{\rm nl}) / t_{\rm sc}\big)$")
  axs[0].set_xlim([ x_min, x_max ])
  axs[1].set_xlim([ x_min, x_max ])
  axs[0].set_ylim([ y0_min, y0_max ])
  axs[1].set_ylim([ y1_min, y1_max ])
  axs[0].axvline(x=0, color="black", ls=":", lw=1.5)
  axs[1].axvline(x=0, color="black", ls=":", lw=1.5)

  x_values = numpy.linspace(-2, 2, 100)
  plot_data.plot_wo_scaling_axis(
    ax       = axs[0],
    x_values = x_values,
    y_values = -2 + 3 * x_values,
    ls       = "--",
    lw       = 1.5
  )
  plot_data.plot_wo_scaling_axis(
    ax       = axs[1],
    x_values = x_values,
    y_values = 1 - x_values,
    ls       = "--",
    lw       = 1.5
  )

  rotation_bounds0 = (x_min, x_max, y0_min, y0_max)
  rotation0 = fit_data.get_line_angle_in_box(
    slope = 3,
    domain_bounds = rotation_bounds0,
    domain_aspect_ratio = 6/4,
  )
  add_annotations.add_text(
    ax          = axs[0],
    x_pos       = 0.06,
    y_pos       = 0.175,
    label       = r"growth regulated by energy \,flux",
    x_alignment = "left",
    y_alignment = "bottom",
    rotate_deg  = rotation0
  )
  add_annotations.add_text(
    ax          = axs[0],
    x_pos       = 0.3,
    y_pos       = 0.215,
    label       = r"$10^{-2}\, \mathcal{M}^3$",
    x_alignment = "center",
    y_alignment = "center",
    rotate_deg  = rotation0
  )

  rotation_bounds1 = (x_min, x_max, y1_min, y1_max)
  rotation1 = fit_data.get_line_angle_in_box(
    slope = -1,
    domain_bounds = rotation_bounds1,
    domain_aspect_ratio = 6/4,
  )
  add_annotations.add_text(
    ax          = axs[1],
    x_pos       = 0.19,
    y_pos       = 0.96,
    label       = r"universal duration",
    x_alignment = "left",
    y_alignment = "top",
    rotate_deg  = rotation1
  )
  add_annotations.add_text(
    ax          = axs[1],
    x_pos       = 0.0825,
    y_pos       = 0.725,
    label       = r"$10 \,\mathcal{M}^{-1} \sim 10 \,t_0 / t_{\rm sc}$",
    x_alignment = "left",
    y_alignment = "top",
    rotate_deg  = rotation1
  )

  add_color.add_cbar_from_cmap(
    ax=axs[0],
    cmap=cmap_Re,
    norm=norm_Re,
    label=r"$\log_{10}(\mathrm{Re})$",
    side="top",
    cbar_padding=0.015,
    fontsize = 22
  )
  add_annotations.add_custom_legend(
    ax           = axs[0],
    artists      = ["o", "s", "D"],
    labels       = [ r"$288^3$", r"$576^3$", r"$1152^3$" ],
    colors       = ["k"] * 3,
    marker_size  = 8,
    line_width   = 1.5,
    fontsize     = 16,
    text_color   = "k",
    position     = "upper left",
    anchor       = (-0.035, 1.0),
    num_cols     = 2,
    text_padding = 0.0
  )
  plot_manager.save_figure(fig, f"nl_scalings.png")


if __name__ == "__main__":
  main()
