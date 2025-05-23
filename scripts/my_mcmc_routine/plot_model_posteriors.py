# ###############################################################
# DEPENDENCIES
# ###############################################################

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from jormi.utils import list_utils
from jormi.ww_io import io_manager
from jormi.ww_data import compute_stats
from jormi.ww_plots import plot_manager, add_annotations

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import typing as npt
    from scipy.stats import gaussian_kde
    from matplotlib.axes import Axes

    from scripts.my_mcmc_routine.base_mcmc import BaseMCMCRoutine

# ###############################################################
# HELPER DATA CLASS
# ###############################################################


class KDEProjectionParams(NamedTuple):
    posterior_samples: "npt.NDArray[np.float64]"
    posterior_kde: "gaussian_kde"
    param_ranges: list[tuple[int, int]]
    col_index: int
    row_index: int
    grid_resolution: int = 20
    num_marginal_samples: int = 50


# ###############################################################
# PLOTTING ROUTINE
# ###############################################################


class PlotModelPosteriors:
    def __init__(self, mcmc_routine: "BaseMCMCRoutine") -> None:
        self.output_directory: Path = mcmc_routine.output_directory
        self.routine_name: str = mcmc_routine.routine_name
        self.num_params: int = mcmc_routine.num_params
        self.plot_posterior_kde: bool = mcmc_routine.plot_posterior_kde
        # fitted params
        self.fitted_posterior_samples: npt.NDArray[np.float64] = (
            mcmc_routine.fitted_posterior_samples
        )
        self.fitted_posterior_kde: gaussian_kde = mcmc_routine.fitted_posterior_kde
        self.fitted_param_labels: list[str] = mcmc_routine.fitted_param_labels
        # output params
        self.output_posterior_samples: npt.NDArray[np.float64] = (
            mcmc_routine.output_posterior_samples
        )
        self.output_posterior_kde: gaussian_kde = mcmc_routine.output_posterior_kde
        self.output_param_labels: list[str] = mcmc_routine.output_param_labels

    def plot(self) -> None:
        self._plot_posteriors(
            posterior_samples=self.fitted_posterior_samples,
            posterior_kde=self.fitted_posterior_kde,
            param_labels=self.fitted_param_labels,
            fig_name=f"{self.routine_name}_fitted_posteriors.png",
        )
        output_is_different_from_fitted = not np.array_equal(
            self.fitted_posterior_samples, self.output_posterior_samples
        )
        if output_is_different_from_fitted:
            self._plot_posteriors(
                posterior_samples=self.output_posterior_samples,
                posterior_kde=self.output_posterior_kde,
                param_labels=self.output_param_labels,
                fig_name=f"{self.routine_name}_output_posteriors.png",
            )

    def _plot_posteriors(
        self,
        posterior_samples: "npt.NDArray[np.float64]",
        posterior_kde: "gaussian_kde",
        param_labels: list[str],
        fig_name: str,
    ) -> None:
        fig, axs = plot_manager.create_figure(
            num_cols=self.num_params,
            num_rows=self.num_params,
            axis_shape=(5, 5),
            fig_scale=0.75,
        )
        param_ranges = []
        for row_index in range(self.num_params):
            for col_index in range(self.num_params):
                ax = axs[row_index, col_index]
                if row_index == col_index:
                    param_range = self._plot_pdf(
                        ax, row_index, posterior_samples, param_labels
                    )
                    param_ranges.append(param_range)
                elif row_index > col_index:
                    self._plot_jpdf(ax, row_index, col_index, posterior_samples)
                else:
                    ax.axis("off")
        self._annotate_plot(axs, param_ranges, param_labels)
        if self.plot_posterior_kde:
            self._plot_kde_projections(
                axs, posterior_samples, posterior_kde, param_ranges
            )
        file_path = io_manager.combine_file_path_parts([
            self.output_directory,
            fig_name,
        ])
        plot_manager.save_figure(fig, file_path, verbose=True)

    def _plot_pdf(
        self,
        ax: "Axes",
        param_index: int,
        posterior_samples: "npt.NDArray[np.float64]",
        param_labels: list[str],
    ) -> tuple["npt.NDArray[np.float64]", "npt.NDArray[np.float64]"]:
        bin_centers, estimated_pdf = compute_stats.estimate_pdf(
            values=posterior_samples[:, param_index], num_bins=20
        )
        ax.step(bin_centers, estimated_pdf, where="mid", lw=2.0, color="black")
        p16, p50, p84 = np.percentile(posterior_samples[:, param_index], [16, 50, 84])
        label = f"{param_labels[param_index]} $= {p50:.2f}_{{-{p50 - p16:.2f}}}^{{+{p84 - p50:.2f}}}$"
        add_annotations.add_text(
            ax=ax,
            x_pos=0.5,
            y_pos=1.05,
            label=label,
            x_alignment="center",
            y_alignment="bottom",
        )
        if param_index > 0:
            ax.tick_params(labelleft=False, labelright=True)
        if param_index < posterior_samples.shape[1] - 1:
            ax.set_xticklabels([])
        pdf_threshold = 0.05 * np.max(estimated_pdf)
        index_start = list_utils.find_first_crossing(
            values=estimated_pdf, target=pdf_threshold, direction="rising"
        )
        if index_start is None:
            param_min = bin_centers[0]
            param_max = bin_centers[-1]
        else:
            index_stop = list_utils.find_first_crossing(
                values=estimated_pdf[index_start:],
                target=pdf_threshold,
                direction="falling",
            )
            if index_stop is not None:
                index_stop += index_start
                param_max = bin_centers[index_stop]
            else:
                param_max = bin_centers[-1]
            param_min = bin_centers[index_start]
        range_width = param_max - param_min
        padding = 2.5 / 100 * range_width
        return (param_min - padding, param_max + padding)

    def _plot_jpdf(
        self,
        ax: "Axes",
        row_index: int,
        col_index: int,
        posterior_samples: "npt.NDArray[np.float64]",
    ) -> None:
        bc_rows, bc_cols, jpdf = compute_stats.estimate_jpdf(
            data_x=posterior_samples[:, col_index],
            data_y=posterior_samples[:, row_index],
            num_bins=50,
        )
        ax.imshow(
            jpdf,
            origin="lower",
            aspect="auto",
            cmap="Blues",
            extent=(
                bc_cols[0],
                bc_cols[-1],
                bc_rows[0],
                bc_rows[-1],
            ),
        )

    def _annotate_plot(
        self,
        axs: "npt.NDArray[Any]",
        param_ranges: list[tuple[int, int]],
        param_labels: list[str],
    ) -> None:
        for row_index in range(self.num_params):
            for col_index in range(self.num_params):
                ax = axs[row_index, col_index]
                if col_index > row_index:
                    continue
                if row_index == col_index:
                    ax.set_xlim(param_ranges[row_index][0], param_ranges[row_index][1])
                    if col_index > 0:
                        ax.tick_params(axis="y", labelright=True)
                else:
                    ax.set_xlim(param_ranges[col_index][0], param_ranges[col_index][1])
                    ax.set_ylim(param_ranges[row_index][0], param_ranges[row_index][1])
                    if col_index == 0:
                        ax.set_ylabel(param_labels[row_index])
                    if col_index > 0:
                        ax.set_yticklabels([])
                if row_index == self.num_params - 1:
                    ax.set_xlabel(param_labels[col_index])
                else:
                    ax.set_xticklabels([])

    def _plot_kde_projections(
        self,
        axs: "npt.NDArray[Any]",
        posterior_samples: "npt.NDArray[np.float64]",
        posterior_kde: "gaussian_kde",
        param_ranges: list[tuple[int, int]],
    ) -> None:
        for row_index in range(self.num_params):
            for col_index in range(self.num_params):
                if col_index >= row_index:
                    continue
                print(f"Estimating KDE projection for axs[{row_index}][{col_index}]")
                params = KDEProjectionParams(
                    posterior_samples=posterior_samples,
                    posterior_kde=posterior_kde,
                    param_ranges=param_ranges,
                    col_index=col_index,
                    row_index=row_index,
                )
                col_grid, row_grid, kde_density = self._compute_2d_kde_projection(
                    params
                )
                axs[row_index, col_index].contour(
                    col_grid,
                    row_grid,
                    kde_density,
                    levels=5,
                    colors="red",
                    linewidths=1.5,
                    alpha=0.75,
                )

    def _compute_2d_kde_projection(
        self, params: KDEProjectionParams
    ) -> tuple[
        "npt.NDArray[np.int64]",
        "npt.NDArray[np.int64]",
        "npt.NDArray[np.float64]",
    ]:
        marginalized_param_indices = [
            param_index
            for param_index in range(self.num_params)
            if param_index not in {params.col_index, params.row_index}
        ]
        col_values = np.linspace(
            params.param_ranges[params.col_index][0],
            params.param_ranges[params.col_index][1],
            params.grid_resolution,
        )
        row_values = np.linspace(
            params.param_ranges[params.row_index][0],
            params.param_ranges[params.row_index][1],
            params.grid_resolution,
        )
        col_grid, row_grid = np.meshgrid(col_values, row_values)
        col_flat = col_grid.ravel()
        row_flat = row_grid.ravel()
        num_grid_points = col_flat.shape[0]
        marginal_sample_indices = np.random.choice(
            params.posterior_samples.shape[0],
            size=params.num_marginal_samples,
            replace=False,
        )
        marginalized_samples = params.posterior_samples[marginal_sample_indices][
            :, marginalized_param_indices
        ]
        grid_points = np.zeros((
            num_grid_points * params.num_marginal_samples,
            self.num_params,
        ))
        grid_points[:, params.col_index] = np.repeat(
            col_flat, params.num_marginal_samples
        )
        grid_points[:, params.row_index] = np.repeat(
            row_flat, params.num_marginal_samples
        )
        grid_points[:, marginalized_param_indices] = np.tile(
            marginalized_samples, (num_grid_points, 1)
        )
        kde_values = params.posterior_kde(grid_points.T)
        kde_density = (
            kde_values.reshape(num_grid_points, params.num_marginal_samples)
            .mean(axis=1)
            .reshape(params.grid_resolution, params.grid_resolution)
        )
        return col_grid, row_grid, kde_density


# END OF MODULE

