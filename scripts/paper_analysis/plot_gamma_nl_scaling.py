import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color

MCMC_MODEL = "linear"
# MCMC_MODEL = "quadratic"
# MCMC_MODEL = "free"

def extract_key_param_samples(fitted_posterior_samples):
  num_params = fitted_posterior_samples.shape[1]
  init_energy_samples    = 10**fitted_posterior_samples[:,0]
  sat_energy_samples     = 10**fitted_posterior_samples[:,1]
  gamma_exp_samples      = fitted_posterior_samples[:,2]
  nl_start_time_samples  = fitted_posterior_samples[:,3]
  sat_start_time_samples = fitted_posterior_samples[:,4]
  if   MCMC_MODEL == "linear":    exponent_samples = 1.0
  elif MCMC_MODEL == "quadratic": exponent_samples = 2.0
  elif MCMC_MODEL == "free":      exponent_samples = fitted_posterior_samples[:,5]
  else: raise ValueError("model makes no sense.")
  nl_start_energy        = init_energy_samples * numpy.exp(gamma_exp_samples * nl_start_time_samples)
  gamma_nl_samples = (sat_energy_samples - nl_start_energy) / (sat_start_time_samples - nl_start_time_samples)**exponent_samples
  return gamma_nl_samples

def main():
  base_directory = Path("/scratch/jh2/nk7952/kriel2025_nl_data/").resolve()
  directories = io_manager.ItemFilter(
    include_string = ["Mach", "Re", "Pm", "Nres"]
  ).filter(
    directory = base_directory
  )
  fig, ax = plot_manager.create_figure()
  cmap_Re, norm_Re = add_color.create_cmap(
    cmap_name = "Blues",
    vmin      = numpy.log10(100),
    vmax      = numpy.log10(5000),
  )
  # ax_inset = add_annotations.add_inset_axis(
  #   ax           = ax,
  #   bounds       = (0.075, 0.6, 0.375, 0.35),
  #   x_label      = r"$\mathrm{Re}$",
  #   fontsize     = 20,
  #   x_label_side = "bottom",
  #   y_label_side = "right",
  # )
  for directory in directories:
    sim_data_path = io_manager.combine_file_path_parts([ directory, "dataset.json" ])
    sim_data_dict = json_files.read_json_file_into_dict(sim_data_path, verbose=False)
    fit_data_path = io_manager.combine_file_path_parts([ directory, f"{MCMC_MODEL}_better_binning", f"stage2_{MCMC_MODEL}_fitted_posterior_samples.npy" ])
    if not io_manager.does_file_exist(fit_data_path): continue
    print(f"Loading: {directory}")
    fitted_posterior_samples = numpy.load(fit_data_path)
    gamma_nl_samples = extract_key_param_samples(fitted_posterior_samples)
    if gamma_nl_samples is None: continue
    Mach_values = sim_data_dict["measured_data"]["rms_Mach_values"]
    Mach_p16, Mach_p50, Mach_p84 = numpy.percentile(numpy.log10(Mach_values), [16, 50, 84])
    Mach_err_lower = Mach_p50 - Mach_p16
    Mach_err_upper = Mach_p84 - Mach_p50
    Re_number = sim_data_dict["plasma_params"]["target_Re"]
    gamma_nl_p16, gamma_nl_p50, gamma_nl_p84 = numpy.percentile(numpy.log10(gamma_nl_samples), [16, 50, 84])
    gamma_nl_err_lower = gamma_nl_p50 - gamma_nl_p16
    gamma_nl_err_upper = gamma_nl_p84 - gamma_nl_p50
    Re_color = cmap_Re(norm_Re(numpy.log10(Re_number)))
    if Re_number < 1000: continue
    ax.errorbar(
      Mach_p50,
      gamma_nl_p50,
      xerr = [
        [Mach_err_lower],
        [Mach_err_upper],
      ],
      yerr = [
        [gamma_nl_err_lower],
        [gamma_nl_err_upper],
      ],
      fmt="o", color=Re_color, mec="black", markersize=5, capsize=3, zorder=3
    )
  ax.set_xlabel(r"$\log_{10}(\mathcal{M})$")
  ax.set_ylabel(r"$\log_{10}(\gamma_{\rm nl})$")
  # ax.set_xlim([-1.5, 1])
  # ax.set_ylim([-6, -0.5])
  ax.axvline(x=0, color="black", ls=":", lw=1.5)
  x_values = numpy.linspace(-2, 2, 100)
  plot_data.plot_wo_scaling_axis(
    ax       = ax,
    x_values = x_values,
    y_values = -2 + 3 * x_values,
    ls       = "--",
    lw       = 1.5
  )
  plot_data.plot_wo_scaling_axis(
    ax       = ax,
    x_values = x_values,
    y_values = -2 + x_values,
    ls       = "-",
    lw       = 1.5
  )
  rotation1 = fit_data.get_line_angle_in_box(
    slope               = 3,
    domain_bounds       = (-1.5, 1.0, -6, -0.5),
    domain_aspect_ratio = 6/4,
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.175,
    y_pos       = 0.325,
    label       = r"$10^{-2}\, \mathcal{M}^3$",
    x_alignment = "center",
    y_alignment = "center",
    rotate_deg  = rotation1
  )
  rotation2 = fit_data.get_line_angle_in_box(
    slope               = 1,
    domain_bounds       = (-1.5, 1.0, -6, -0.5),
    domain_aspect_ratio = 6/4,
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.175,
    y_pos       = 0.625,
    label       = r"$10^{-2}\, \mathcal{M}$",
    x_alignment = "center",
    y_alignment = "center",
    rotate_deg  = rotation2
  )
  add_color.add_cbar_from_cmap(
    ax    = ax,
    cmap  = cmap_Re,
    norm  = norm_Re,
    label = r"$\log_{10}(\mathrm{Re})$",
    side  = "top",
  )
  plot_manager.save_figure(fig, f"gamma_nl_scaling_{MCMC_MODEL}_better_binning.png")


if __name__ == "__main__":
  main()


## end