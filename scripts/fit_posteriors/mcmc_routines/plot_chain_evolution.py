## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any, final

## third-party
import numpy

## personal
from jormi.ww_io import manage_io
from jormi.ww_plots import annotate_axis, manage_plots
from jormi.ww_types import box_positions

##
## === PLOTTING ROUTINE
##


@final
class PlotChainEvolution:

    def __init__(
        self,
        mcmc_routine: Any,
    ) -> None:
        self.num_params = mcmc_routine.num_params
        self.num_walkers = mcmc_routine.num_walkers
        self.raw_chain = mcmc_routine.raw_chain
        self.labels = mcmc_routine.fitted_param_labels
        self.routine_name = mcmc_routine.routine_name
        self.output_directory = mcmc_routine.output_directory

    def plot(
        self,
    ) -> None:
        fig, axs = manage_plots.create_figure(
            num_rows=self.num_params,
            num_cols=1,
            share_x=True,
        )
        axs = axs[:, 0]
        for param_index in range(self.num_params):
            for walker_index in range(self.num_walkers):
                axs[param_index].plot(
                    self.raw_chain[:, walker_index, param_index],
                    alpha=0.3,
                    lw=0.5,
                    zorder=3,
                )
            p16, p50, p84 = numpy.percentile(self.raw_chain[:, :, param_index], [16, 50, 84])
            axs[param_index].axhspan(p16, p84, color="black", alpha=0.25, lw=1.5, zorder=4)
            axs[param_index].axhline(p50, color="black", ls=":", lw=1.5, zorder=5)
            axs[param_index].set_ylabel(self.labels[param_index])
        annotate_axis.add_text(
            ax=axs[0],
            x_pos=0.95,
            y_pos=0.05,
            label=f"{self.num_walkers} walkers",
            x_alignment=box_positions.MPLPositions.Align.Side.Right,
            y_alignment=box_positions.MPLPositions.Align.Side.Bottom,
        )
        axs[-1].set_xlabel("steps")
        fig_name = f"{self.routine_name}_fitted_chain_evolution.png"
        file_path = self.output_directory / fig_name
        manage_plots.save_figure(
            fig=fig,
            fig_path=file_path,
        )


## } MODULE
