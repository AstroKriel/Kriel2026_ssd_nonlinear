# ###############################################################
# DEPENDENCIES
# ###############################################################

from typing import TYPE_CHECKING

import numpy as np
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, add_annotations

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import typing as npt

    from scripts.my_mcmc_routine.base_mcmc import BaseMCMCRoutine


# ###############################################################
# PLOTTING ROUTINE
# ###############################################################
class PlotChainEvolution:
    def __init__(self, mcmc_routine: "BaseMCMCRoutine") -> None:
        self.num_params: int = mcmc_routine.num_params
        self.num_walkers: int = mcmc_routine.num_walkers
        self.raw_chain: npt.NDArray[np.float64] = mcmc_routine.raw_chain
        self.labels: list[str] = mcmc_routine.fitted_param_labels
        self.routine_name: str = mcmc_routine.routine_name
        self.output_directory: Path = mcmc_routine.output_directory

    def plot(self) -> None:
        fig, axs = plot_manager.create_figure(
            num_rows=self.num_params, num_cols=1, share_x=True
        )
        for param_index in range(self.num_params):
            for walker_index in range(self.num_walkers):
                axs[param_index].plot(
                    self.raw_chain[:, walker_index, param_index],
                    alpha=0.3,
                    lw=0.5,
                    zorder=3,
                )
            p16, p50, p84 = np.percentile(
                self.raw_chain[:, :, param_index], [16, 50, 84]
            )
            axs[param_index].axhspan(
                p16, p84, color="black", alpha=0.25, lw=1.5, zorder=4
            )
            axs[param_index].axhline(p50, color="black", ls=":", lw=1.5, zorder=5)
            axs[param_index].set_ylabel(self.labels[param_index])
        add_annotations.add_text(
            ax=axs[0],
            x_pos=0.95,
            y_pos=0.05,
            label=f"{self.num_walkers} walkers",
            x_alignment="right",
            y_alignment="bottom",
        )
        axs[-1].set_xlabel("steps")
        fig_name = f"{self.routine_name}_fitted_chain_evolution.png"
        file_path = io_manager.combine_file_path_parts([
            self.output_directory,
            fig_name,
        ])
        plot_manager.save_figure(fig, file_path, verbose=True)


# END OF MODULE

