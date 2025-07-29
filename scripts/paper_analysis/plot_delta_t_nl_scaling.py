import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color


def extract_key_param_samples(fitted_posterior_samples):
  start_nl_time_samples  = fitted_posterior_samples[:,1]
  start_sat_time_samples = fitted_posterior_samples[:,2]
  return start_sat_time_samples - start_nl_time_samples

def main():
  mcmc_model = "free"
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
  for directory in directories:
    sim_data_path = io_manager.combine_file_path_parts([ directory, "dataset.json" ])
    sim_data_dict = json_files.read_json_file_into_dict(sim_data_path, verbose=False)
    fit_data_path = io_manager.combine_file_path_parts([ directory, mcmc_model, f"stage2_{mcmc_model}_fitted_posterior_samples.npy" ])
    if not io_manager.does_file_exist(fit_data_path): continue
    print(f"Loading: {directory}")
    fitted_posterior_samples = numpy.load(fit_data_path)
    delta_t_samples = extract_key_param_samples(fitted_posterior_samples)
    delta_t_p16, delta_t_p50, delta_t_p84 = numpy.percentile(numpy.log10(delta_t_samples), [16, 50, 84])
    delta_t_err_lower = delta_t_p50 - delta_t_p16
    delta_t_err_upper = delta_t_p84 - delta_t_p50
    Mach_values = sim_data_dict["raw_data"]["Mach_values"]
    Mach_p16, Mach_p50, Mach_p84 = numpy.percentile(numpy.log10(Mach_values), [16, 50, 84])
    Mach_err_lower = Mach_p50 - Mach_p16
    Mach_err_upper = Mach_p84 - Mach_p50
    Mach_number = sim_data_dict["plasma_params"]["Mach"]
    Re_number = sim_data_dict["plasma_params"]["Re"]
    Re_values = float(Re_number) / float(Mach_number) * numpy.array(Mach_values)
    Re_p16, Re_p50, Re_p84 = numpy.percentile(Re_values, [16, 50, 84])
    Re_err_lower = Re_p50 - Re_p16
    Re_err_upper = Re_p84 - Re_p50
    Re_color = cmap_Re(norm_Re(numpy.log10(Re_p50)))
    if Re_p50 < 1000: continue
    ax.errorbar(
      Mach_p50,
      delta_t_p50,
      xerr = [
        [Mach_err_lower],
        [Mach_err_upper],
      ],
      yerr = [
        [delta_t_err_lower],
        [delta_t_err_upper],
      ],
      fmt="o", color=Re_color, mec="black", markersize=5, capsize=3, zorder=3
    )
  # ax.set_xscale("log")
  # ax.set_yscale("log")
  # ax.set_xlim([1e-1, 1e1])
  # ax.set_ylim([6e-1, 200])
  ax.set_xlim([-1.5, 1.0])
  ax.set_ylim([0.0, 2.5])
  x_values = numpy.linspace(-2, 2, 100)
  plot_data.plot_wo_scaling_axis(
    ax       = ax,
    x_values = x_values,
    y_values = 1 - x_values,
    ls       = "--",
    lw       = 1.5
  )
  rotation = fit_data.get_line_angle_in_box(
    slope               = -1,
    domain_bounds       = (-1.5, 1.0, 0.0, 2.5),
    domain_aspect_ratio = 6/4,
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.25,
    y_pos       = 0.825,
    label       = r"$\mathcal{M}^{-1}$",
    x_alignment = "center",
    y_alignment = "center",
    rotate_deg  = rotation
  )
  ax.set_xlabel(r"$\log_{10}(\mathcal{M}) \equiv \langle u^2 \rangle^{1/2} / c_s$")
  ax.set_ylabel(r"$\log_{10}(t_{\rm sat} - t_{\rm nl})$")
  add_color.add_cbar_from_cmap(
    ax    = ax,
    cmap  = cmap_Re,
    norm  = norm_Re,
    label = r"$\log_{10}(\ell_0 \langle u^2 \rangle^{1/2} / \nu)$",
    side  = "top",
  )
  ax.axvline(x=0, color="black", ls=":", lw=1.5)
  plot_manager.save_figure(fig, f"delta_t_scaling_{mcmc_model}.png")


if __name__ == "__main__":
  main()


## end