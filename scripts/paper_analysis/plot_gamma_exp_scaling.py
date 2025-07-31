import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color


def extract_key_param_samples(fitted_posterior_samples):
  gamma_samples = fitted_posterior_samples[:,2]
  return gamma_samples

def main():
  # mcmc_model = "linear"
  # mcmc_model = "quadratic"
  # mcmc_model = "free"
  base_directory = Path("/scratch/jh2/nk7952/kriel2025_nl_data/").resolve()
  directories = io_manager.ItemFilter(
    include_string = ["Mach", "Re", "Pm", "Nres"]
  ).filter(
    directory = base_directory
  )
  fig, ax = plot_manager.create_figure()
  cmap_Mach, norm_Mach = add_color.create_cmap(
    cmap_name = "cmr.watermelon",
    vmin      = numpy.log10(0.1),
    vmid      = 0,
    vmax      = numpy.log10(5),
  )
  for directory in directories:
    sim_data_path = io_manager.combine_file_path_parts([ directory, "dataset.json" ])
    sim_data_dict = json_files.read_json_file_into_dict(sim_data_path, verbose=False)
    fit_data_path = io_manager.combine_file_path_parts([ directory, f"{mcmc_model}_better_binning", f"stage2_{mcmc_model}_fitted_posterior_samples.npy" ])
    if not io_manager.does_file_exist(fit_data_path): continue
    print(f"Loading: {directory}")
    fitted_posterior_samples = numpy.load(fit_data_path)
    gamma_samples = extract_key_param_samples(fitted_posterior_samples)
    gamma_p16, gamma_p50, gamma_p84 = numpy.percentile(gamma_samples, [16, 50, 84])
    gamma_err_lower = numpy.log10(gamma_p50) - numpy.log10(gamma_p16)
    gamma_err_upper = numpy.log10(gamma_p84) - numpy.log10(gamma_p50)
    Mach_values = sim_data_dict["measured_data"]["rms_Mach_values"]
    Mach_number = sim_data_dict["plasma_params"]["target_Mach"]
    Re_number = sim_data_dict["plasma_params"]["target_Re"]
    Re_values = float(Re_number) / float(Mach_number) * numpy.array(Mach_values)
    Re_p16, Re_p50, Re_p84 = numpy.percentile(Re_values, [16, 50, 84])
    Re_err_lower = numpy.log10(Re_p50) - numpy.log10(Re_p16)
    Re_err_upper = numpy.log10(Re_p84) - numpy.log10(Re_p50)
    Mach_color = cmap_Mach(norm_Mach(numpy.log10(Mach_number)))
    if Re_p50 < 1000: continue
    ax.errorbar(
      numpy.log10(Re_p50),
      numpy.log10(gamma_p50 / Mach_number),
      xerr = [
        [Re_err_lower],
        [Re_err_upper],
      ],
      yerr = [
        [gamma_err_lower / Mach_number],
        [gamma_err_upper / Mach_number],
      ],
      fmt="o", color=Mach_color, mec="black", markersize=5, capsize=3, zorder=-numpy.log10(Mach_number)
    )
  ax.set_xlim([3, 3.8])
  ax.set_ylim([-0.4, 0.5])
  x_values = numpy.linspace(3, 4, 100)
  plot_data.plot_wo_scaling_axis(
    ax       = ax,
    x_values = x_values,
    y_values = numpy.log10(3.5e-2) + 1/2 * x_values,
    ls       = "--",
    lw       = 1.5,
    zorder   = -15
  )
  plot_data.plot_wo_scaling_axis(
    ax       = ax,
    x_values = x_values,
    y_values = numpy.log10(4.5e-2) + 1/3 * x_values,
    ls       = ":",
    lw       = 1.5,
    zorder   = -15
  )
  rotation1 = fit_data.get_line_angle_in_box(
    slope               = 1/2,
    domain_bounds       = (3, 3.8, -0.4, 0.5),
    domain_aspect_ratio = 6/4,
  )
  rotation2 = fit_data.get_line_angle_in_box(
    slope               = 1/3,
    domain_bounds       = (3, 3.8, -0.4, 0.5),
    domain_aspect_ratio = 6/4,
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.25,
    y_pos       = 0.725,
    label       = r"$3 \times 10^{-2}\, \mathrm{Re}^{1/2}$",
    x_alignment = "center",
    y_alignment = "center",
    rotate_deg  = rotation1
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.775,
    y_pos       = 0.4,
    label       = r"$5 \times 10^{-2}\, \mathrm{Re}^{1/3}$",
    x_alignment = "center",
    y_alignment = "center",
    rotate_deg  = rotation2
  )
  ax.set_xlabel(r"$\log_{10}(\mathrm{Re}) \equiv \log_{10}(\ell_0 \langle u^2 \rangle^{1/2} / \nu)$")
  ax.set_ylabel(r"$\log_{10}(\gamma_{\rm exp} / \langle u^2 \rangle^{1/2})$")
  add_color.add_cbar_from_cmap(
    ax    = ax,
    cmap  = cmap_Mach,
    norm  = norm_Mach,
    label = r"$\log_{10}(\langle u^2 \rangle^{1/2} / c_s)$",
    side  = "top",
  )
  plot_manager.save_figure(fig, f"gamma_exp_scaling_{mcmc_model}_better_binning.png")


if __name__ == "__main__":
  main()


## end