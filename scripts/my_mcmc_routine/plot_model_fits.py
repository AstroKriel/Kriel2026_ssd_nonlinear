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


class PlotModelFits:
    def __init__(self, mcmc_routine: "BaseMCMCRoutine", num_curves: int = 100) -> None:
        self.num_curves: int = num_curves
        self.routine_name: str = mcmc_routine.routine_name
        self.output_directory: Path = mcmc_routine.output_directory
        self.x_values: npt.NDArray[np.float64] = mcmc_routine.x_values
        self.y_values: npt.NDArray[np.float64] = mcmc_routine.y_values
        self.data_label: str | None = mcmc_routine.data_label
        self.num_params: int = mcmc_routine.num_params
        self.fitted_posterior_samples: npt.NDArray[np.float64] = (
            mcmc_routine.fitted_posterior_samples
        )
        self.model_func: Callable[
            [tuple[float, ...] | npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ] = mcmc_routine._model
        self._annotate_fitted_params: Callable[[list[Axes]], None] = (
            mcmc_routine._annotate_fitted_params
        )
        self._annotate_output_params: Callable[[list[Axes]], None] = (
            mcmc_routine._annotate_output_params
        )

    def plot(self) -> None:
        fig, axs = plot_manager.create_figure(
            num_rows=3, share_x=True
        )  # TODO(AstroKriel): Fix create_figure type signature
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
        fig_file_path = io_manager.combine_file_path_parts([
            self.output_directory,
            fig_name,
        ])
        plot_manager.save_figure(fig, fig_file_path, verbose=True)

    def _plot_data(self, axs: tuple["Axes", "Axes"]) -> None:
        dy_dx_values = np.gradient(self.y_values, self.x_values)
        style: dict[str, Any] = {
            "color": "blue",
            "marker": "o",
            "ms": 5,
            "ls": "-",
            "lw": 1.0,
            "zorder": 3,
        }
        axs[0].plot(self.x_values, self.y_values, **style)
        axs[1].plot(self.x_values, dy_dx_values, **style)
        axs[1].axhline(y=0.0, color="black", ls="--", lw=1.5, zorder=0)

    def _plot_model(self, axs: tuple["Axes", "Axes"]) -> None:
        random_selector = np.random.default_rng(seed=42)
        random_indices = random_selector.choice(
            self.num_params, size=min(self.num_curves, self.num_params), replace=False
        )
        modelled_curves = np.array([
            self.model_func(self.fitted_posterior_samples[random_index]).squeeze()
            for random_index in random_indices
        ])
        p16, p50, p84 = np.percentile(modelled_curves, [16, 50, 84], axis=0)
        axs[0].plot(self.x_values, p50, color="red", lw=2, zorder=4)
        axs[0].fill_between(self.x_values, p16, p84, color="red", alpha=0.25, zorder=3)

    def _plot_residuals(self, axs: tuple["Axes", "Axes", "Axes"]) -> None:
        median_params = np.median(self.fitted_posterior_samples, axis=0)
        modelled_y = self.model_func(median_params).squeeze()
        y_residuals = self.y_values - modelled_y
        axs[2].plot(
            self.x_values,
            y_residuals,
            color="red",
            marker="o",
            ms=5,
            ls="-",
            lw=1.0,
            zorder=3,
        )
        axs[2].axhline(y=0, color="black", ls="--", lw=1.5, zorder=0)


# END OF MODULE
