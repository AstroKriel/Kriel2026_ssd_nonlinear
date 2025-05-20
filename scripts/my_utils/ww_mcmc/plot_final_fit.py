## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from jormi.ww_plots import plot_manager
from jormi.ww_io import io_manager


## ###############################################################
## PLOTTING ROUTINE
## ###############################################################

class PlotFinalFit:

  def __init__(self, mcmc_routine, num_curves: int = 100):
    self.mcmc_routine = mcmc_routine
    self.num_curves   = num_curves

  def plot(self):
    fig, axs = plot_manager.create_figure(num_rows=2, share_x=True)
    self._plot_data(axs)
    self._plot_model(axs)
    fig_name = f"final_fit.png"
    fig_file_path = io_manager.combine_file_path_parts([self.mcmc_routine.output_directory, fig_name])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.mcmc_routine.verbose)

  def _plot_data(self, axs):
    style = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    axs[0].plot(self.mcmc_routine.x_values, numpy.log10(self.mcmc_routine.y_values), **style)
    axs[1].plot(self.mcmc_routine.x_values, self.mcmc_routine.y_values, **style)
    axs[1].axhline(y=0, color="black", ls="--", lw=1.5, zorder=0)

  def _plot_model(self, axs):
    rng = numpy.random.default_rng(seed=42)
    random_indices = rng.choice(
      self.mcmc_routine.num_params,
      size    = min(self.num_curves, self.mcmc_routine.num_params),
      replace = False
    )
    modelled_curves = numpy.array([
      self.mcmc_routine._model(self.mcmc_routine.fitted_posterior_samples[sample_index]).squeeze()
      for sample_index in random_indices
    ])
    p16, p50, p84 = numpy.percentile(modelled_curves, [16, 50, 84], axis=0)
    axs[0].plot(self.mcmc_routine.x_values, numpy.log10(p50), color="red", lw=2, zorder=4)
    axs[0].fill_between(self.mcmc_routine.x_values, numpy.log10(p16), numpy.log10(p84), color="red", alpha=0.25, zorder=3)
    axs[1].plot(self.mcmc_routine.x_values, p50, color="red", lw=2, zorder=4)
    axs[1].fill_between(self.mcmc_routine.x_values, p16, p84, color="red", alpha=0.25, zorder=3)


## END OF MODULE