# START OF MODULE


# ###############################################################
# DEPENDENCIES
# ###############################################################

from typing import TYPE_CHECKING, Any

import numpy as np
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy import typing as npt
    from matplotlib.axes import Axes

    from scripts.my_mcmc_routine.base_mcmc import BaseMCMCRoutine

# ###############################################################
# PLOTTING ROUTINE
# ###############################################################


class PlotFinalFits:
    def __init__(self, mcmc_routine: "BaseMCMCRoutine", num_curves: int = 100) -> None:
        self.num_curves: int = num_curves
        self.output_directory: Path = mcmc_routine.output_directory
        self.x_values: npt.NDArray[np.float64] = mcmc_routine.x_values
        self.y_values: npt.NDArray[np.float64] = mcmc_routine.y_values
        self.num_params: int = mcmc_routine.num_params
        self.model_func: Callable[
            [tuple[float, ...] | npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ] = mcmc_routine._model  # TODO(AstroKriel): If you're going to be using anything from a class outside of the class itself, don't prepend with _ (_model -> model)
        # Alternatively, why not just call self.mcmc_routine.whatever?
        self.fitted_posterior_samples: npt.NDArray[np.float64] = (
            mcmc_routine.fitted_posterior_samples
        )

    def plot(self) -> None:
        fig, axs = plot_manager.create_figure(num_rows=2, share_x=True)
        self._plot_data(axs)
        self._plot_model(axs)
        fig_name = "final_fit.png"
        fig_file_path = io_manager.combine_file_path_parts([
            self.output_directory,
            fig_name,
        ])
        plot_manager.save_figure(fig, fig_file_path, verbose=True)

    def _plot_data(self, axs: tuple["Axes", "Axes"]) -> None:
        # dict[str, Any] indicates that this can be considered kwargs
        style: dict[str, Any] = {
            "color": "blue",
            "marker": "o",
            "ms": 5,
            "ls": "-",
            "lw": 1.0,
            "zorder": 3,
        }
        axs[0].plot(self.x_values, np.log10(self.y_values), **style)
        axs[1].plot(self.x_values, self.y_values, **style)
        axs[1].axhline(y=0, color="black", ls="--", lw=1.5, zorder=0)

    def _plot_model(self, axs: tuple["Axes", "Axes"]) -> None:
        rng = np.random.default_rng(seed=42)
        random_indices = rng.choice(
            self.num_params, size=min(self.num_curves, self.num_params), replace=False
        )
        modelled_curves = np.array([
            self.model_func(self.fitted_posterior_samples[sample_index]).squeeze()
            for sample_index in random_indices
        ])
        p16, p50, p84 = np.percentile(modelled_curves, [16, 50, 84], axis=0)
        axs[0].plot(self.x_values, np.log10(p50), color="red", lw=2, zorder=4)
        axs[0].fill_between(
            self.x_values,
            np.log10(p16),
            np.log10(p84),
            color="red",
            alpha=0.25,
            zorder=3,
        )
        axs[1].plot(self.x_values, p50, color="red", lw=2, zorder=4)
        axs[1].fill_between(self.x_values, p16, p84, color="red", alpha=0.25, zorder=3)

    def _label_plot(self, axs: tuple["Axes", "Axes"]) -> None:
        axs[0].set_ylabel(r"$\log_{10}(E_{\mathrm{mag}})$")
        axs[1].set_ylabel(r"$E_{\mathrm{mag}}$")
        axs[1].set_xlabel(r"time")


# END OF MODULE
