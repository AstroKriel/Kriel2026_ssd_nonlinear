import re
import numpy
from jormi.ww_io import io_manager, json_files
from jormi.ww_plots import plot_manager, plot_data, add_annotations


def extract_re_mach(filename):
    match = re.search(r"Re(?P<Re>\d+)Mach(?P<Mach>[\d.]+)", filename)
    if match:
      Re = int(match.group("Re"))
      Mach = float(match.group("Mach"))
      return Re, Mach
    else: return None, None


def get_alpha(fit_params):
  (log10_init_energy, _, gamma) = fit_params["stage1_params"]
  (start_nl_time, start_sat_time, log10_sat_energy) = fit_params["stage2_params"]
  init_energy     = 10**log10_init_energy
  start_nl_energy = init_energy * numpy.exp(gamma * start_nl_time)
  sat_energy      = 10**log10_sat_energy
  alpha           = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
  return alpha


def plot_powerlaw_passing_through(ax, domain_bounds, coordinate, slope, num_samples, ls):
  x_values = numpy.linspace(domain_bounds[0], domain_bounds[1], num_samples)
  (x1, y1) = coordinate
  a0 = y1 / x1**slope
  y_values = a0 * x_values**slope
  plot_data.plot_wo_scaling_axis(
    ax     = ax,
    x_values = x_values,
    y_values = y_values,
    ls     = ls,
    lw     = 2
  )


def main():
  directory  = "fits"
  io_manager.
  file_names = io_manager.filter_files(
    directory      = directory,
    include_string = "_params",
    suffix         = ".json",
  )
  fig, axs = plot_manager.create_figure(num_cols=2)
  for file_name in file_names:
    file_path  = io_manager.combine_file_path_parts([ directory, file_name ])
    Re, Mach   = extract_re_mach(file_name)
    fit_params = json_files.read_json_file_into_dict(file_path, verbose=0)
    alpha      = get_alpha(fit_params)
    if Mach < 1: color = "blue"
    else: color = "red"
    axs[0].plot(Re, alpha, color=color, marker="o", ms=5)
    axs[1].plot(Mach, alpha, color=color, marker="o", ms=5)
  axs[0].set_xscale("log")
  axs[0].set_yscale("log")
  axs[1].set_xscale("log")
  axs[1].set_yscale("log")
  axs[0].set_xlabel(r"Re")
  axs[0].set_ylabel(r"$\alpha$")
  axs[1].set_xlabel(r"$\mathcal{M}$")
  axs[1].set_yticklabels([])
  plot_powerlaw_passing_through(
    ax            = axs[1],
    domain_bounds = (1e-2, 1e1),
    coordinate    = (1, 1e-3),
    slope         = 2,
    num_samples   = 10,
    ls            = ":"
  )
  plot_powerlaw_passing_through(
    ax            = axs[1],
    domain_bounds = (1e-2, 1e1),
    coordinate    = (1, 1e-3),
    slope         = 3,
    num_samples   = 10,
    ls            = "--",
  )
  add_annotations.add_text(
    ax    = axs[1],
    x_pos = 0.05,
    y_pos = 0.4,
    label = r"$\sim\mathcal{M}^2$",
  )
  add_annotations.add_text(
    ax    = axs[1],
    x_pos = 0.2,
    y_pos = 0.15,
    label = r"$\sim\mathcal{M}^3$",
  )
  plot_manager.save_figure(fig, "test.png")


if __name__ == "__main__":
  main()


## end