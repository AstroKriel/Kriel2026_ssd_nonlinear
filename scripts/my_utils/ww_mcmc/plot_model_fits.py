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

  def __init__(
      self,
      mcmc_routine,
      num_curves : int = 100,
    ):
    self.mcmc_routine  = mcmc_routine
    self.num_curves    = num_curves
    self.y_data_label  = self.mcmc_routine.y_data_label
    self._annotate_fitted_params = self.mcmc_routine._annotate_fitted_params

  def plot(self):
    fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
    self._plot_data(axs)
    self._plot_model(axs)
    self._plot_residuals(axs)
    self._annotate_fitted_params(axs)
    if self.y_data_label is not None:
      axs[0].set_ylabel(self.y_data_label)
      stripped_y_data_label = self.y_data_label.strip("$")
      axs[1].set_ylabel(r"$\frac{d}{dt}\," + stripped_y_data_label + "$")
    axs[2].set_ylabel(r"median residuals")
    axs[2].set_xlabel(r"time")
    fig_name = f"{self.mcmc_routine.routine_name}_fit.png"
    fig_file_path = io_manager.combine_file_path_parts([self.mcmc_routine.output_directory, fig_name])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.mcmc_routine.verbose)

  def _plot_data(self, axs):
    dy_dx_values = numpy.gradient(self.mcmc_routine.y_values, self.mcmc_routine.x_values)
    style = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    axs[0].plot(self.mcmc_routine.x_values, self.mcmc_routine.y_values, **style)
    axs[1].plot(self.mcmc_routine.x_values, dy_dx_values, **style)
    axs[1].axhline(y=0, color="black", ls="--", lw=1.5, zorder=0)

  def _plot_model(self, axs):
    rng = numpy.random.default_rng(seed=42)
    random_indices = rng.choice(
      self.mcmc_routine.num_params,
      size    = min(self.num_curves, self.mcmc_routine.num_params),
      replace = False
    )
    modelled_curves = numpy.array([
      self.mcmc_routine._model(self.mcmc_routine.fitted_posterior_samples[sample_index])
      for sample_index in random_indices
    ])
    p16, p50, p84 = numpy.percentile(modelled_curves, [16, 50, 84], axis=0)
    axs[0].plot(self.mcmc_routine.x_values, p50, color="red", lw=2, zorder=4)
    axs[0].fill_between(self.mcmc_routine.x_values, p16, p84, color="red", alpha=0.25, zorder=3)

  def _plot_residuals(self, axs):
    median_params = numpy.median(self.mcmc_routine.fitted_posterior_samples, axis=0)
    model_y       = self.mcmc_routine._model(median_params)
    residuals     = self.mcmc_routine.y_values - model_y
    axs[2].plot(self.mcmc_routine.x_values, residuals, color="red", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    axs[2].axhline(y=0, color="black", ls="--", lw=1.5, zorder=0)


## END OF MODULE