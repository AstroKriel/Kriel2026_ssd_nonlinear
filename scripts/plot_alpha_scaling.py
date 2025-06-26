import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color


def extract_key_param_samples(fitted_posterior_samples):
  init_energy_samples    = 10**fitted_posterior_samples[:, 0]
  sat_energy_samples     = 10**fitted_posterior_samples[:, 1]
  gamma_samples          = fitted_posterior_samples[:, 2]
  start_nl_time_samples  = fitted_posterior_samples[:, 3]
  start_sat_time_samples = fitted_posterior_samples[:, 4]
  start_nl_energy        = init_energy_samples * numpy.exp(gamma_samples * start_nl_time_samples)
  alpha_samples          = (sat_energy_samples - start_nl_energy) / (start_sat_time_samples - start_nl_time_samples)
  return gamma_samples, alpha_samples, sat_energy_samples

def main():
  base_directory = Path("../data/").resolve()
  data_directories = io_manager.ItemFilter(
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
  ax_inset = add_annotations.add_inset_axis(
    ax           = ax,
    bounds       = (0.075, 0.6, 0.375, 0.35),
    x_label      = r"$\mathrm{Re}$",
    fontsize     = 20,
    x_label_side = "bottom",
    y_label_side = "right",
  )
  for data_directory in data_directories:
    sim_data_path = io_manager.combine_file_path_parts([ data_directory, "dataset.json" ])
    sim_data_dict = json_files.read_json_file_into_dict(sim_data_path, verbose=False)
    fit_data_path = io_manager.combine_file_path_parts([ data_directory, "stage2_fitted_posterior_samples.npy" ])
    if not io_manager.does_file_exist(fit_data_path): continue
    print(f"Loading: {data_directory}")
    fitted_posterior_samples = numpy.load(fit_data_path)
    _, alpha_samples, _ = extract_key_param_samples(fitted_posterior_samples)
    alpha_p16, alpha_p50, alpha_p84 = numpy.percentile(alpha_samples, [16, 50, 84])
    alpha_err_lower = alpha_p50 - alpha_p16
    alpha_err_upper = alpha_p84 - alpha_p50
    Mach_number = sim_data_dict["plasma_params"]["Mach"]
    Re_number = sim_data_dict["plasma_params"]["Re"]
    Re_color = cmap_Re(norm_Re(numpy.log10(Re_number)))
    ax.errorbar(
      Mach_number,
      alpha_p50,
      yerr = [
        [alpha_err_lower],
        [alpha_err_upper],
      ],
      fmt="o", color=Re_color, markersize=5, capsize=3, zorder=3
    )
    ax_inset.errorbar(
      Re_number,
      alpha_p50,
      yerr = [
        [alpha_err_lower],
        [alpha_err_upper],
      ],
      fmt="o", color=Re_color, markersize=5, capsize=3, zorder=3
    )
  ax.set_xlabel(r"$\langle u^2 \rangle^{1/2}_{\mathcal{V}, \mathrm{kin}} / c_s$")
  ax.set_ylabel(r"$\alpha$")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim([0.01, 10])
  ax.set_ylim([1e-7, 1])
  ax_inset.set_xscale("log")
  ax_inset.set_yscale("log")
  ax_inset.set_xlim([100, 1e4])
  ax_inset.set_ylim([1e-7, 1])
  ax_inset.set_yticklabels([])
  ax.axvline(x=1, color="black", ls="--", lw=1.5)
  x_values = numpy.logspace(-3, 2, 100)
  # coefficient = fit_data.get_powerlaw_coefficient(exponent=3, x_ref=1e-1, y_ref=1e-5)
  log10_coefficient = fit_data.get_linear_intercept(slope=3, x_ref=-1, y_ref=-5)
  print(10**log10_coefficient)
  plot_data.plot_wo_scaling_axis(
    ax       = ax,
    x_values = x_values,
    y_values = 10**log10_coefficient * x_values**3,
    ls       = "--",
    lw       = 1.5
  )
  plot_data.plot_wo_scaling_axis(
    ax       = ax,
    x_values = x_values,
    y_values = 0.08 * x_values**3,
    color    = "red",
    ls       = "--",
    lw       = 1.5
  )
  rotation = fit_data.get_line_angle_in_box(
    slope               = 3,
    domain_bounds       = (-2, 1, -7, 0),
    domain_aspect_ratio = 6/4,
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.1,
    y_pos       = 0.05,
    label       = r"$10^{-2}\, \mathcal{M}^3$",
    x_alignment = "left",
    y_alignment = "bottom",
    rotate_deg  = rotation
  )
  add_color.add_cbar_from_cmap(
    ax    = ax,
    cmap  = cmap_Re,
    norm  = norm_Re,
    label = r"$\log_{10}(\mathrm{Re})$",
    side  = "top",
  )
  plot_manager.save_figure(fig, "alpha_scaling.png")


if __name__ == "__main__":
  main()


## end