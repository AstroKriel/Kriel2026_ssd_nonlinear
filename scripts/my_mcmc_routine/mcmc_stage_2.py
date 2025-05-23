# ###############################################################
# DEPENDENCIES
# ###############################################################

from typing import TYPE_CHECKING, Literal, override

import numpy as np

from . import base_mcmc

if TYPE_CHECKING:
    from types import FunctionType
    from pathlib import Path

    from numpy import typing as npt
    from scipy.stats import gaussian_kde
    from matplotlib.axes import Axes

# ###############################################################
# HELPER FUNCTION
# ###############################################################


# TODO(AstroKriel): If this is identical to the one in  stage_1, why not move it into base_mcmc?
def plot_param_percentiles(
    ax: "Axes",
    samples: "npt.NDArray[np.float64]",
    orientation: Literal["h", "horizontal", "v", "vertical"],
) -> None:
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])
    if "h" in orientation.lower():
        ax_line = ax.axhline
        ax_span = ax.axhspan
    elif "v" in orientation.lower():
        ax_line = ax.axvline
        ax_span = ax.axvspan
    else:
        msg = "`orientation` must either be `horizontal` (`h`) or `vertical` (`v`)."
        raise ValueError(msg)
    ax_line(p50, color="green", ls=":", lw=1.5, zorder=5)
    ax_span(p16, p84, color="green", ls="-", lw=1.5, alpha=0.3, zorder=4)


# ###############################################################
# STAGE 2 MCMC FITTER
# ###############################################################


class MCMCStage2Routine(base_mcmc.BaseMCMCRoutine):
    def __init__(
        self,
        *,
        output_directory: "str | Path",
        x_values: "list[float] | npt.NDArray[np.float64]",
        y_values: "list[float] | npt.NDArray[np.float64]",
        initial_params: tuple[float, ...],
        prior_kde: "gaussian_kde | None" = None,
        likelihood_sigma: float = 1.0,
        plot_posterior_kde: bool = False,
    ) -> None:
        self.max_time: npt.NDArray[np.float64] = np.max(x_values)
        super().__init__(
            routine_name="stage2",
            output_directory=output_directory,
            x_values=x_values,
            y_values=y_values,
            initial_params=initial_params,
            prior_kde=prior_kde,
            likelihood_sigma=likelihood_sigma,
            plot_posterior_kde=plot_posterior_kde,
            data_label=r"$E_{\mathrm{mag}}$",
            fitted_param_labels=[
                r"$\log_{10}(E_{\mathrm{init}})$",
                r"$\log_{10}(E_{\mathrm{sat}})$",
                r"$\gamma$",
                r"$t_{\mathrm{nl}}$",
                r"$t_{\mathrm{sat}}$",
            ],
        )

    @override
    def _model(
        self, param_vectors: "tuple[float, ...] | npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.float64]":
        # TODO(AstroKriel): More comments to describe some of the more intricate transformations
        param_vectors = np.atleast_2d(param_vectors)  # (N, P)
        # output dimensions
        num_local_walkers = param_vectors.shape[0]  # N
        num_data_points = len(self.x_values)  # T
        # unpack model parameters (P = 5)
        log10_init_energy, log10_sat_energy, gamma, start_nl_time, start_sat_time = (
            param_vectors.T
        )
        # reshape parameters to allow for vectorising over param-rows
        x_values_2d = self.x_values[None, :]  # shape (1, T)
        start_nl_time_2d = start_nl_time[:, None]  # shape (N, 1)
        start_sat_time_2d = start_sat_time[:, None]  # shape (N, 1)
        gamma_2d = gamma[:, None]  # shape (N, 1)
        # mask SSD phases
        mask_exp_phase = x_values_2d < start_nl_time_2d
        mask_nl_phase = (start_nl_time_2d <= x_values_2d) & (
            x_values_2d < start_sat_time_2d
        )
        mask_sat_phase = start_sat_time_2d < x_values_2d
        # compute model constants (per walker)
        init_energy = 10**log10_init_energy  # (N,)
        sat_energy = 10**log10_sat_energy  # (N,)
        start_nl_energy = init_energy * np.exp(gamma * start_nl_time)  # (N,)
        alpha = (sat_energy - start_nl_energy) / (
            start_sat_time - start_nl_time
        )  # (N,)
        init_energy_2d = init_energy[:, None]  # (N, 1)
        sat_energy_2d = sat_energy[:, None]  # (N, 1)
        start_nl_energy_2d = start_nl_energy[:, None]  # (N, 1)
        alpha_2d = alpha[:, None]  # (N, 1)
        # assemble modelled SSD phases
        energy = np.zeros((num_local_walkers, num_data_points))
        energy[mask_exp_phase] = (init_energy_2d * np.exp(gamma_2d * x_values_2d))[
            mask_exp_phase
        ]  # (N, T)
        energy[mask_nl_phase] = (
            start_nl_energy_2d + alpha_2d * (x_values_2d - start_nl_time_2d)
        )[mask_nl_phase]  # (N, T)
        energy[mask_sat_phase] = np.broadcast_to(
            sat_energy_2d, (num_local_walkers, num_data_points)
        )[mask_sat_phase]  # (N, T)
        return energy

    @override
    def _get_valid_params_mask(
        self, param_vectors: "tuple[float, ...] | npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.bool]":
        param_vectors = np.atleast_2d(param_vectors)
        num_local_walkers = param_vectors.shape[0]
        log10_init_energy, log10_sat_energy, gamma, start_nl_time, start_sat_time = (
            param_vectors.T
        )
        valid_log10_init_energy = (log10_init_energy > -30) & (log10_init_energy < -5)
        valid_log10_sat_energy = (log10_sat_energy > -5) & (log10_sat_energy < 0)
        valid_gamma = (gamma > 0) & (gamma < 2)
        valid_start_nl_time = (0.1 * self.max_time < start_nl_time) & (
            start_nl_time < start_sat_time
        )
        valid_start_sat_time = start_sat_time < self.max_time
        valid_params_mask = (
            valid_log10_init_energy
            & valid_log10_sat_energy
            & valid_gamma
            & valid_start_nl_time
            & valid_start_sat_time
        )
        if num_local_walkers == 1:
            return valid_params_mask[0]
        return valid_params_mask

    @override
    def _get_kde_params(
        self, param_vectors: "npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.float64]":
        # ignore the transition times; use a unifrom prior for them
        return np.asarray(param_vectors[:, :3])

    @override
    def _annotate_fitted_params(self, axs: list["Axes"]) -> None:
        init_energy_samples = 10 ** self.fitted_posterior_samples[:, 0]
        sat_energy_samples = 10 ** self.fitted_posterior_samples[:, 1]
        gamma_samples = self.fitted_posterior_samples[:, 2]
        start_nl_time_samples = self.fitted_posterior_samples[:, 3]
        start_sat_time_samples = self.fitted_posterior_samples[:, 4]
        start_nl_energy_samples = init_energy_samples * np.exp(
            gamma_samples * start_nl_time_samples
        )
        alpha_samples = (sat_energy_samples - start_nl_energy_samples) / (
            start_sat_time_samples - start_nl_time_samples
        )
        plot_param_percentiles(axs[0], sat_energy_samples, orientation="horizontal")
        plot_param_percentiles(axs[1], alpha_samples, orientation="horizontal")
        for row_index in range(len(axs)):
            plot_param_percentiles(
                axs[row_index], start_nl_time_samples, orientation="vertical"
            )
            plot_param_percentiles(
                axs[row_index], start_sat_time_samples, orientation="vertical"
            )

    # TODO(AstroKriel): Override all abstract methods
    @override
    def _annotate_output_params(self, axs: list["Axes"]) -> None:
        log10_sat_energy_samples = self.output_posterior_samples[:, 1]
        plot_param_percentiles(
            axs[0], log10_sat_energy_samples, orientation="horizontal"
        )


# END OF MODULE
