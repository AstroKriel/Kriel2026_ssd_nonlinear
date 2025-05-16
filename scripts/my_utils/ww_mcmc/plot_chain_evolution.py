## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from jormi.ww_plots import plot_manager, add_annotations
from jormi.ww_io import io_manager
from . import base_plotter


## ###############################################################
## PLOTTING ROUTINE
## ###############################################################
class PlotChainEvolution(base_plotter.BaseMCMCPlotter):

  def plot(self):
    num_params  = self.mcmc_routine.num_params
    num_walkers = self.mcmc_routine.num_walkers
    raw_chain   = self.mcmc_routine.raw_chain # (walkers, steps, params)
    labels      = self.mcmc_routine.param_labels
    fig, axs = plot_manager.create_figure(num_rows=num_params, num_cols=1, share_x=True)
    for param_index in range(num_params):
      for walker_index in range(num_walkers):
        axs[param_index].plot(raw_chain[:, walker_index, param_index], alpha=0.3, lw=0.5, zorder=3)
      p16, p50, p84 = numpy.percentile(raw_chain[:, :, param_index], [16, 50, 84])
      axs[param_index].axhspan(p16, p84, color="black", alpha=0.25, lw=1.5, zorder=4)
      axs[param_index].axhline(p50, color="black", linestyle=":", lw=1.5, zorder=5)
      axs[param_index].set_ylabel(labels[param_index])
    add_annotations.add_text(
      ax          = axs[0],
      x_pos       = 0.95,
      y_pos       = 0.05,
      label       = f"{num_walkers} walkers",
      x_alignment = "right",
      y_alignment = "bottom",
    )
    axs[-1].set_xlabel("steps")
    fig_name  = f"{self.mcmc_routine.routine_name}_chain_evolution.png"
    file_path = io_manager.combine_file_path_parts([ self.mcmc_routine.output_directory, fig_name ])
    plot_manager.save_figure(fig, file_path, verbose=self.mcmc_routine.verbose)


## END OF MODULE