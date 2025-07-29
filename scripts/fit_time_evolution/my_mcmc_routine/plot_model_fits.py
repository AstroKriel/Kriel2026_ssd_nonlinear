## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from jormi.ww_plots import plot_manager
from jormi.ww_io import io_manager


## ###############################################################
## PLOTTING ROUTINE
## ###############################################################

class PlotModelFits:

  def __init__(self, mcmc_routine, num_curves: int = 100):
    self.num_curves               = num_curves
    self.routine_name             = mcmc_routine.routine_name
    self.output_directory         = mcmc_routine.output_directory
    self.x_values                 = mcmc_routine.x_values
    self.y_ave_values             = mcmc_routine.y_values
    self.y_std_values             = mcmc_routine.likelihood_sigma
    self.data_label               = mcmc_routine.data_label
    self.num_params               = mcmc_routine.num_params
    self.fitted_posterior_samples = mcmc_routine.fitted_posterior_samples
    self.model_func               = mcmc_routine._model
    self._annotate_fitted_params  = mcmc_routine._annotate_fitted_params
    self._annotate_output_params  = mcmc_routine._annotate_output_params

  def plot(self):
    fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
    self._plot_data(axs)
    self._plot_model(axs)
    self._plot_residuals(axs)
    self._annotate_fitted_params(axs)
    self._annotate_output_params(axs)
    if self.data_label is not None:
      axs[0].set_ylabel(self.data_label)
      stripped_data_label = self.data_label.strip("$")
      axs[1].set_ylabel(r"$\frac{d}{dt}\," + stripped_data_label + "$")
    axs[2].set_ylabel(r"median residuals")
    axs[2].set_xlabel(r"time")
    fig_name = f"{self.routine_name}_fit.png"
    fig_file_path = io_manager.combine_file_path_parts([self.output_directory, fig_name])
    plot_manager.save_figure(fig, fig_file_path, verbose=True)

  def _plot_data(self, axs):
    dy_dx_values = numpy.gradient(self.y_ave_values, self.x_values)
    style = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    axs[0].errorbar(
      self.x_values,
      self.y_ave_values,
      yerr = self.y_std_values,
      fmt="o", color="blue", markersize=4, elinewidth=1, capsize=2, zorder=7
    )
    axs[1].plot(self.x_values, dy_dx_values, **style)
    axs[1].axhline(y=0.0, color="black", ls="--", lw=1.5, zorder=0)

  def _plot_model(self, axs):
    random_selector = numpy.random.default_rng(seed=42)
    num_samples     = self.fitted_posterior_samples.shape[0]
    random_indices  = random_selector.choice(
      num_samples,
      size=min(self.num_curves, num_samples),
      replace=False
    )
    modelled_curves = []
    for random_index in random_indices:
      params = self.fitted_posterior_samples[random_index]
      y_model = self.model_func(params).squeeze()
      modelled_curves.append(y_model)
      axs[0].plot(self.x_values, y_model, color="gray", alpha=0.2, lw=0.5, zorder=1)
    modelled_curves = numpy.array(modelled_curves)
    p16, p84 = numpy.percentile(modelled_curves, [16, 84], axis=0)
    axs[0].fill_between(self.x_values, p16, p84, color="red", alpha=0.25, zorder=2)


  def _plot_residuals(self, axs):
    median_params = numpy.median(self.fitted_posterior_samples, axis=0)
    modelled_y    = self.model_func(median_params).squeeze()
    y_residuals   = self.y_ave_values - modelled_y
    axs[2].plot(self.x_values, y_residuals, color="red", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    axs[2].axhline(y=0, color="black", ls="--", lw=1.5, zorder=0)



## END OF MODULE