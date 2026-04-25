## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any, Callable, NamedTuple, final

## third-party
import numpy
from numpy.typing import NDArray

## personal
from jormi import ww_lists
from jormi.ww_io import manage_io
from jormi.ww_arrays import compute_array_stats
from jormi.ww_plots import annotate_axis, manage_plots
from jormi.ww_types import box_positions

##
## === HELPER DATA CLASS
##


class KDEProjectionParams(
        NamedTuple, ):
    posterior_samples: NDArray[Any]
    posterior_kde: Callable[[NDArray[Any]], NDArray[Any]]
    param_ranges: list[tuple[float, float]]
    col_index: int
    row_index: int
    grid_resolution: int = 20
    num_marginal_samples: int = 50


##
## === PLOTTING ROUTINE
##


@final
class PlotModelPosteriors:

    def __init__(
        self,
        mcmc_routine: Any,
    ) -> None:
        self.output_directory = mcmc_routine.output_directory
        self.routine_name = mcmc_routine.routine_name
        self.num_params = mcmc_routine.num_params
        self.plot_posterior_kde = mcmc_routine.plot_posterior_kde
        self.show_progress = mcmc_routine.show_progress
        ## fitted params
        self.fitted_posterior_samples = mcmc_routine.fitted_posterior_samples
        self.fitted_posterior_kde = mcmc_routine.fitted_posterior_kde
        self.fitted_param_labels = mcmc_routine.fitted_param_labels
        ## output params
        self.output_posterior_samples = mcmc_routine.output_posterior_samples
        self.output_posterior_kde = mcmc_routine.output_posterior_kde
        self.output_param_labels = mcmc_routine.output_param_labels

    def plot(
        self,
    ) -> None:
        self._plot_posteriors(
            posterior_samples=self.fitted_posterior_samples,
            posterior_kde=self.fitted_posterior_kde,
            param_labels=self.fitted_param_labels,
            fig_name=f"{self.routine_name}_fitted_posteriors.png",
        )
        output_is_different_from_fitted = not numpy.array_equal(
            self.fitted_posterior_samples,
            self.output_posterior_samples,
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
        posterior_samples: NDArray[Any],
        posterior_kde: Callable[[NDArray[Any]], NDArray[Any]],
        param_labels: list[str],
        fig_name: str,
    ) -> None:
        fig, axs = manage_plots.create_figure(
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
                    param_range = self._plot_pdf(ax, row_index, posterior_samples, param_labels)
                    param_ranges.append(param_range)
                elif row_index > col_index:
                    self._plot_jpdf(ax, row_index, col_index, posterior_samples)
                else:
                    ax.axis("off")
        self._annotate_plot(axs, param_ranges, param_labels)
        if self.plot_posterior_kde:
            self._plot_kde_projections(axs, posterior_samples, posterior_kde, param_ranges)
        file_path = manage_io.combine_file_path_parts([self.output_directory, fig_name])
        manage_plots.save_figure(fig, file_path, verbose=True)

    def _plot_pdf(
        self,
        ax: Any,
        param_index: int,
        posterior_samples: NDArray[Any],
        param_labels: list[str],
    ) -> tuple[float, float]:
        pdf_result = compute_array_stats.estimate_pdf(
            values=posterior_samples[:, param_index],
            num_bins=20,
        )
        bin_centers = pdf_result.bin_centers
        estimated_pdf = pdf_result.densities
        ax.step(bin_centers, estimated_pdf, where="mid", lw=2.0, color="black")
        p16, p50, p84 = numpy.percentile(posterior_samples[:, param_index], [16, 50, 84])
        label = f"{param_labels[param_index]} $= {p50:.2f}_{{-{p50-p16:.2f}}}^{{+{p84-p50:.2f}}}$"
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.5,
            y_pos=0.98,
            label=label,
            x_alignment=box_positions.MPLPositions.Align.Center.Center,
            y_alignment=box_positions.MPLPositions.Align.Side.Top,
        )
        if param_index > 0:
            ax.tick_params(labelleft=False, labelright=True)
        if param_index < posterior_samples.shape[1] - 1:
            ax.set_xticklabels([])
        pdf_threshold = 0.05 * numpy.max(estimated_pdf)
        index_start = ww_lists.get_index_of_first_crossing(
            values=[float(value) for value in estimated_pdf],
            target=pdf_threshold,
            direction="rising",
        )
        if index_start is None:
            param_min = bin_centers[0]
            param_max = bin_centers[-1]
        else:
            index_stop = ww_lists.get_index_of_first_crossing(
                values=[float(value) for value in estimated_pdf[index_start:]],
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
        padded_range = (
            param_min - padding,
            param_max + padding,
        )
        return padded_range

    def _plot_jpdf(
        self,
        ax: Any,
        row_index: int,
        col_index: int,
        posterior_samples: NDArray[Any],
    ) -> None:
        jpdf_result = compute_array_stats.estimate_jpdf(
            data_x=posterior_samples[:, col_index],
            data_y=posterior_samples[:, row_index],
            num_bins=50,
        )
        bc_rows = jpdf_result.row_centers
        bc_cols = jpdf_result.col_centers
        jpdf = jpdf_result.densities
        ax.imshow(
            jpdf,
            origin="lower",
            aspect="auto",
            cmap="Blues",
            extent=[
                bc_cols[0],
                bc_cols[-1],
                bc_rows[0],
                bc_rows[-1],
            ],
        )

    def _annotate_plot(
        self,
        axs: Any,
        _param_ranges: list[tuple[float, float]],
        param_labels: list[str],
    ) -> None:
        for row_index in range(self.num_params):
            for col_index in range(self.num_params):
                ax = axs[row_index, col_index]
                if col_index > row_index:
                    continue
                if row_index == col_index:
                    # ax.set_xlim(param_ranges[row_index][0], param_ranges[row_index][1]) # debug
                    if col_index > 0:
                        ax.tick_params(axis="y", labelright=True)
                else:
                    # ax.set_xlim(param_ranges[col_index][0], param_ranges[col_index][1]) # debug
                    # ax.set_ylim(param_ranges[row_index][0], param_ranges[row_index][1]) # debug
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
        axs: Any,
        posterior_samples: NDArray[Any],
        posterior_kde: Callable[[NDArray[Any]], NDArray[Any]],
        param_ranges: list[tuple[float, float]],
    ) -> None:
        for row_index in range(self.num_params):
            for col_index in range(self.num_params):
                if col_index >= row_index:
                    continue
                if self.show_progress:
                    print(f"Estimating KDE projection for axs[{row_index}][{col_index}]")
                params = KDEProjectionParams(
                    posterior_samples=posterior_samples,
                    posterior_kde=posterior_kde,
                    param_ranges=param_ranges,
                    col_index=col_index,
                    row_index=row_index,
                )
                col_grid, row_grid, kde_density = self._compute_2d_kde_projection(params)
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
        self,
        params: KDEProjectionParams,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        marginalized_param_indices = [
            param_index for param_index in range(self.num_params)
            if (param_index != params.col_index) and (param_index != params.row_index)
        ]
        col_values = numpy.linspace(
            params.param_ranges[params.col_index][0],
            params.param_ranges[params.col_index][1],
            params.grid_resolution,
        )
        row_values = numpy.linspace(
            params.param_ranges[params.row_index][0],
            params.param_ranges[params.row_index][1],
            params.grid_resolution,
        )
        col_grid, row_grid = numpy.meshgrid(col_values, row_values)
        col_flat = col_grid.ravel()
        row_flat = row_grid.ravel()
        num_grid_points = col_flat.shape[0]
        marginal_sample_indices = numpy.random.choice(
            params.posterior_samples.shape[0],
            size=params.num_marginal_samples,
            replace=False,
        )
        marginalized_samples = params.posterior_samples[marginal_sample_indices][
            :,
            marginalized_param_indices,
        ]
        grid_points = numpy.zeros((
            num_grid_points * params.num_marginal_samples,
            self.num_params,
        ), )
        grid_points[:, params.col_index] = numpy.repeat(col_flat, params.num_marginal_samples)
        grid_points[:, params.row_index] = numpy.repeat(row_flat, params.num_marginal_samples)
        grid_points[:, marginalized_param_indices] = numpy.tile(
            marginalized_samples,
            (num_grid_points, 1),
        )
        kde_values = params.posterior_kde(grid_points.T)
        kde_density = kde_values.reshape(num_grid_points, params.num_marginal_samples).mean(
            axis=1,
        ).reshape(params.grid_resolution, params.grid_resolution)
        return col_grid, row_grid, kde_density


## } MODULE
