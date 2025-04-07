## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from scipy.ndimage import gaussian_filter1d as scipy_filter1d
from loki.ww_plots import plot_manager
from loki.ww_data import interpolate_data


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def generate_uniform_domain(num_points):
  x_data = numpy.linspace(0, 100, num_points)
  return x_data

def generate_nonuniform_domain(num_points):
  x_uniform = generate_uniform_domain(num_points)
  numpy.random.seed(0)
  x_perturbs = numpy.random.uniform(-5, 5, size=num_points)
  x_data = numpy.sort(x_uniform + x_perturbs)
  x_min, x_max = x_data[0], x_data[-1]
  x_data = 101 * (x_data - x_min) / (x_max - x_min)
  return x_data

def generate_nonuniform_data(x_data, noise_level, axs=None):
  growth_rate  = 10
  x_transition = 40
  if axs is not None:
    axs[0].axvline(x=x_transition, ls="--", color="black", zorder=1)
    axs[1].axvline(x=x_transition, ls="--", color="black", zorder=1)
    axs[2].axvline(x=x_transition, ls="--", color="black", zorder=1)
  y_data = numpy.piecewise(
    x_data,
    [
      x_data < x_transition,
      x_data >= x_transition
    ],
    [
      lambda x: growth_rate * x,
      growth_rate * x_transition
    ]
  )
  numpy.random.seed(42)
  y_data += 0.5 * growth_rate * numpy.random.normal(0, noise_level, x_data.shape)
  return y_data

def compute_derivatives(x, y):
  dy_dx   = numpy.gradient(y, x)
  d2y_dx2 = numpy.gradient(dy_dx, x)
  return dy_dx, d2y_dx2

def plot_data(axs, x_data, y_data, color, label):
  plot_style = { "color":color, "ls":"-", "lw":1.5, "marker":"o", "ms":5, "zorder":3, "label":label }
  dydx_data, d2ydx2_data = compute_derivatives(x_data, y_data)
  axs[0].plot(x_data, y_data,      **plot_style)
  axs[1].plot(x_data, dydx_data,   **plot_style)
  axs[2].plot(x_data, d2ydx2_data, **plot_style)


## ###############################################################
## DEMO: DERIVATIVE COMPARISONS
## ###############################################################
def main():
  fig, axs = plot_manager.create_figure(num_rows=3, share_x=True, axis_shape=(6, 10))
  num_points_raw    = 100
  num_points_interp = 50
  x_data_raw        = generate_nonuniform_domain(num_points_raw)
  y_data_raw        = generate_nonuniform_data(x_data=x_data_raw, noise_level=3.0, axs=axs)
  _x_data_interp    = generate_uniform_domain(num_points_interp)
  x_data_interp, y_data_interp = interpolate_data.interpolate_1d(x_data_raw, y_data_raw, _x_data_interp, kind="linear")
  axs[0].plot(x_data_raw, y_data_raw, color="blue", marker="o", ms=5, zorder=3, label="raw data")
  plot_data(
    axs    = axs,
    x_data = x_data_interp,
    y_data = y_data_interp,
    color  = "red",
    label  = "uniformly sampled",
  )
  plot_data(
    axs    = axs,
    x_data = x_data_interp,
    y_data = scipy_filter1d(y_data_interp, 1.0),
    color  = "green",
    label  = "uniformly sampled + smoothed",
  )
  axs[0].set_ylabel("y-values")
  axs[1].set_ylabel("first derivatives")
  axs[2].set_ylabel("second derivatives")
  axs[2].set_xlabel("x-values")
  axs[0].legend(loc="lower right")
  axs[1].axhline(y=0, ls="--", color="black", zorder=1)
  axs[2].axhline(y=0, ls="--", color="black", zorder=1)
  axs[1].set_ylim([-20, 20])
  axs[2].set_ylim([-5, 5])
  plot_manager.save_figure(fig, "estimate_ssd_phases_using_gradients.png")



## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT