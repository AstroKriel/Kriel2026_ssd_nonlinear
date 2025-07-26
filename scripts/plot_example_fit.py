## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from pathlib import Path
from scipy.stats import levy
from jormi.ww_plots import plot_manager
from jormi.ww_io import io_manager, json_files
import my_mcmc_routine


## ###############################################################
## PLOTTING ROUTINE
## ###############################################################

def plot_param_percentiles(ax, samples, orientation):
  p16, p50, p84 = numpy.percentile(samples, [16, 50, 84])
  if   "h" in orientation.lower():
    ax_line = ax.axhline
    ax_span = ax.axhspan
  elif "v" in orientation.lower():
    ax_line = ax.axvline
    ax_span = ax.axvspan
  else: raise ValueError("`orientation` must either be `horizontal` (`h`) or `vertical` (`v`).")
  ax_line(p50, color="green", ls="--", lw=1.5, zorder=10)
  ax_span(p16, p84, color="green", ls="-", lw=1.5, alpha=0.15, zorder=0)

class PlotFinalFits:

  def __init__(self, sim_name, num_curves: int = 100):
    self.data_directory = Path(f"/scratch/jh2/nk7952/kriel2025_nl_data/{sim_name}").resolve()
    data_path = io_manager.combine_file_path_parts([ self.data_directory, "dataset.json" ])
    data_dict = json_files.read_json_file_into_dict(data_path)
    stage2_fitted_posterior_path = io_manager.combine_file_path_parts([ self.data_directory, "stage2_fitted_posterior_samples.npy" ])
    # self.fitted_posterior_samples = numpy.load(stage2_fitted_posterior_path)
    # self.num_params = self.fitted_posterior_samples.shape[1]
    self.x_values   = data_dict["raw_data"]["time"]
    self.y_values   = data_dict["raw_data"]["magnetic_energy"]
    self.num_curves = num_curves
    # stage2_mcmc = my_mcmc_routine.MCMCStage2Routine(
    #   output_directory   = self.data_directory,
    #   x_values           = self.x_values,
    #   y_values           = self.y_values,
    #   initial_params     = tuple(
    #     numpy.median(self.fitted_posterior_samples[:,param_index])
    #     for param_index in range(self.num_params)
    #   )
    # )
    # self.model_func = stage2_mcmc._model

  def plot(self):
    fig, axs = plot_manager.create_figure(num_rows=2, share_x=True)
    self._plot_data(axs)
    # self._plot_model(axs)
    # self._annotate_fitted_params(axs)
    self._label_plot(axs)
    fig_name = f"example_fit.png"
    fig_file_path = io_manager.combine_file_path_parts([self.data_directory, fig_name])
    plot_manager.save_figure(fig, fig_file_path, verbose=True)

  def _plot_data(self, axs):
    num_bins = 20
    x_bin_edges = numpy.linspace(0, numpy.max(self.x_values), num_bins+1)
    x_bin_centers = 0.5 * (x_bin_edges[1:] + x_bin_edges[:-1])
    x_bin_indices = numpy.digitize(self.x_values, x_bin_edges) - 1
    y_ave_s = numpy.zeros(num_bins)
    y_std_s = numpy.zeros(num_bins)
    log10_y_ave_s = numpy.zeros(num_bins)
    log10_y_std_s = numpy.zeros(num_bins)
    for bin_index in range(num_bins):
      bin_mask = (x_bin_indices == bin_index)
      if numpy.any(bin_mask):
        y_values_in_bin = numpy.array(self.y_values)[bin_mask]
        log10_y_values_in_bin = numpy.log10(y_values_in_bin)
        y_ave_s[bin_index] = numpy.mean(y_values_in_bin)
        y_std_s[bin_index] = numpy.std(y_values_in_bin)
        log10_y_ave_s[bin_index] = numpy.mean(log10_y_values_in_bin)
        log10_y_std_s[bin_index] = numpy.std(log10_y_values_in_bin)
      else:
        y_ave_s[bin_index] = numpy.nan
        y_std_s[bin_index] = numpy.nan
        log10_y_ave_s[bin_index] = numpy.nan
        log10_y_std_s[bin_index] = numpy.nan
    valid_bins = ~numpy.isnan(y_ave_s)
    # axs[1].errorbar(
    #   x_bin_centers[valid_bins],
    #   y_ave_s[valid_bins],
    #   yerr = y_std_s[valid_bins],
    #   fmt="o", color="blue", markersize=4, elinewidth=1, capsize=2, zorder=7
    # )
    axs[0].errorbar(
      x_bin_centers[valid_bins],
      log10_y_ave_s[valid_bins],
      yerr = log10_y_std_s[valid_bins],
      fmt="o", color="blue", markersize=4, elinewidth=1, capsize=2, zorder=7
    )
    t_turb = 0.5 / 5
    axs[1].plot(
      x_bin_centers[valid_bins],
      levy.logpdf(x_bin_centers[valid_bins], loc=7, scale=t_turb),
      color="red", ls="-"
    )
    axs[1].plot(
      x_bin_centers[valid_bins],
      levy.logpdf(x_bin_centers[valid_bins], loc=7, scale=10*t_turb),
      color="red", ls="--"
    )

  def _plot_model(self, axs):
    rng = numpy.random.default_rng(seed=42)
    ## generate random curve-indices
    random_indices = rng.choice(
      self.fitted_posterior_samples.shape[0],
      size=min(self.num_curves, self.fitted_posterior_samples.shape[0]),
      replace=False
    )
    ## compute all model predictions
    all_models = numpy.array([
      self.model_func(sample).squeeze()
      for sample in self.fitted_posterior_samples
    ])
    ## estimate the "best fit" (minimised L2-error)
    errors = numpy.sum(numpy.square(all_models - self.y_values), axis=1)
    best_fit_index = numpy.argmin(errors)
    best_fit_curve = all_models[best_fit_index]
    ## plot random samples
    for idx in random_indices:
      curve = all_models[idx]
      axs[1].plot(self.x_values, curve, color="grey", alpha=0.25, lw=1.0, zorder=3)
      axs[0].plot(self.x_values, numpy.log10(curve), color="grey", alpha=0.25, lw=1.0, zorder=3)
    ## plot percentile region
    p16, p84 = numpy.percentile(all_models, [16, 84], axis=0)
    axs[1].fill_between(self.x_values, p16, p84, color="red", alpha=0.5, zorder=4)
    axs[0].fill_between(self.x_values, numpy.log10(p16), numpy.log10(p84), color="red", alpha=0.5, zorder=4)
    ## plot "best fit"
    axs[1].plot(self.x_values, best_fit_curve, color="black", lw=2.0, zorder=5)
    axs[0].plot(self.x_values, numpy.log10(best_fit_curve), color="black", lw=2.0, zorder=5)

  def _annotate_fitted_params(self, axs):
    start_nl_time_samples   = self.fitted_posterior_samples[:,3]
    start_sat_time_samples  = self.fitted_posterior_samples[:,4]
    for row_index in range(len(axs)):
      plot_param_percentiles(axs[row_index], start_nl_time_samples, orientation="vertical")
      plot_param_percentiles(axs[row_index], start_sat_time_samples, orientation="vertical")

  def _label_plot(self, axs):
    axs[1].set_ylabel(r"$E_{\mathrm{mag}}$")
    axs[0].set_ylabel(r"$\log_{10}(E_{\mathrm{mag}})$")
    axs[1].set_xlabel(r"time")


def main():
  sim_name = "Mach5.0Re1500Pm1Nres576v5"
  PlotFinalFits(sim_name).plot()

if __name__ == "__main__":
  main()


## END OF MODULE