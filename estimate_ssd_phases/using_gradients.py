## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from scipy.ndimage import gaussian_filter1d as scipy_filter1d
from jormi.ww_io import flash_data
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def plot_data(axs, time_data, energy_data, color, label=None):
  plot_style = { "color":color, "ls":"-", "lw":1.5, "marker":"o", "ms":5, "zorder":3, "label":label }
  dydt_data   = numpy.gradient(energy_data, time_data)
  d2ydt2_data = numpy.gradient(dydt_data, time_data)
  axs[0].plot(time_data, energy_data, **plot_style)
  axs[1].plot(time_data, dydt_data,   **plot_style)
  axs[2].plot(time_data, d2ydt2_data, **plot_style)


## ###############################################################
## DEMO: DERIVATIVE COMPARISONS
## ###############################################################
def main():
  fig, axs = plot_manager.create_figure(
    num_rows   = 3,
    num_cols   = 2,
    share_x    = True,
    axis_shape = (6, 10),
    x_spacing  = 0.3,
    y_spacing  = 0.1,
  )
  time_raw, energy_raw = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re1500/Mach0.5/Pm1/576",
    dataset_name = "mag",
    time_start   = 1.0,
  )
  log10_energy_raw = numpy.log10(energy_raw)
  axs[0,0].plot(time_raw, log10_energy_raw, color="blue", marker="o", ms=5, zorder=3, label="raw data")
  axs[0,1].plot(time_raw, energy_raw, color="blue", marker="o", ms=5, zorder=3, label="raw data")
  _time_interp = numpy.linspace(numpy.min(time_raw), numpy.max(time_raw), 100)
  time_interp, log10_energy_interp = interpolate_data.interpolate_1d(time_raw, log10_energy_raw, _time_interp, kind="linear")
  log10_energy_interp_filtered = scipy_filter1d(log10_energy_interp, 2.0)
  plot_data(
    axs         = axs[:,0],
    time_data   = time_interp,
    energy_data = log10_energy_interp_filtered,
    color       = "red",
    label       = r"sampling in $\log_{10}E_{\rm mag}$ + filtering"
  )
  plot_data(
    axs         = axs[:,1],
    time_data   = time_interp,
    energy_data = numpy.power(10, log10_energy_interp_filtered),
    color       = "red",
  )
  axs[0,0].set_ylabel(r"$\log_{10}(E_{\rm mag})$")
  axs[1,0].set_ylabel(r"${\rm d} \log_{10}(E_{\rm mag}) / {\rm d} t$")
  axs[2,0].set_ylabel(r"${\rm d^2} \log_{10}(E_{\rm mag}) / {\rm d} t^2$")
  axs[2,0].set_xlabel(r"$t$")
  axs[1,0].axhline(y=0, ls="--", color="black", zorder=1)
  axs[2,0].axhline(y=0, ls="--", color="black", zorder=1)
  axs[0,0].legend(loc="lower right")
  axs[0,1].set_ylabel(r"$E_{\rm mag}$")
  axs[1,1].set_ylabel(r"${\rm d} E_{\rm mag} / {\rm d} t$")
  axs[2,1].set_ylabel(r"${\rm d^2} E_{\rm mag} / {\rm d} t^2$")
  axs[2,1].set_xlabel(r"$t$")
  axs[1,1].axhline(y=0, ls="--", color="black", zorder=1)
  axs[2,1].axhline(y=0, ls="--", color="black", zorder=1)
  plot_manager.save_figure(fig, "estimate_using_gradients.png")



## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT