# ###############################################################
# DEPENDENCIES
# ###############################################################

from typing import TYPE_CHECKING, Literal, override

import numpy as np

from . import base_mcmc

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import typing as npt
    from matplotlib.axes import Axes

# ###############################################################
# HELPER FUNCTION
# ###############################################################


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
# STAGE 1 MCMC FITTER
# ###############################################################


class MCMCStage1Routine(base_mcmc.BaseMCMCRoutine):
    def __init__(
        self,
        *,
        output_directory: "str | Path",
        x_values: "list[float] | npt.NDArray[np.float64]",
        y_values: "list[float] | npt.NDArray[np.float64]",
        initial_params: tuple[float, ...],
        likelihood_sigma: float = 1.0,
        plot_posterior_kde: bool = False,
    ) -> None:
        self.log10_e: npt.NDArray[np.float64] = np.log10(np.exp(1))
        self.max_time: npt.NDArray[np.float64] = np.max(x_values)
        super().__init__(
            routine_name="stage1",
            output_directory=output_directory,
            x_values=x_values,
            y_values=np.log10(y_values),
            initial_params=initial_params,
            likelihood_sigma=likelihood_sigma,
            plot_posterior_kde=plot_posterior_kde,
            data_label=r"$\log_{10}(E_{\mathrm{mag}})$",
            fitted_param_labels=[
                r"$\log_{10}(E_{\mathrm{init}})$",
                r"$\gamma$",
                r"$t_{\mathrm{approx}}$",
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
        # unpack model parameters (P = 3)
        log10_init_energy, gamma, transition_time = param_vectors.T
        # reshape parameters to allow for vectorising over param-rows
        x_values_2d = self.x_values[None, :]  # shape (1, T)
        gamma_2d = gamma[:, None]  # shape (N, 1)
        transition_time_2d = transition_time[:, None]  # shape (N, 1)
        log10_init_energy_2d = log10_init_energy[:, None]  # shape (N, 1)
        # mask (reduced) SSD phases
        mask_exp_phase = x_values_2d < transition_time_2d
        mask_sat_phase = ~mask_exp_phase
        # compute (reduced) SSD phases
        log10_energy_exp_values = (
            log10_init_energy_2d + self.log10_e * gamma_2d * x_values_2d
        )  # (N, T)
        log10_energy_sat_values = (
            log10_init_energy_2d + self.log10_e * gamma_2d * transition_time_2d
        )  # (N, 1)
        log10_energy_sat_values = np.broadcast_to(
            log10_energy_sat_values, (num_local_walkers, num_data_points)
        )  # (N, T)
        # assemble modelled (reduced) SSD phases
        log10_energy = np.empty((num_local_walkers, num_data_points))
        log10_energy[mask_exp_phase] = log10_energy_exp_values[mask_exp_phase]
        log10_energy[mask_sat_phase] = log10_energy_sat_values[mask_sat_phase]
        return log10_energy

    @override
    def _get_valid_params_mask(
        self, param_vectors: "tuple[float, ...] | npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.bool]":
        param_vectors = np.atleast_2d(param_vectors)
        num_local_walkers = param_vectors.shape[0]
        log10_init_energy, gamma, transition_time = param_vectors.T
        valid_log10_init_energy = (log10_init_energy > -30) & (log10_init_energy < -5)
        valid_gamma = (gamma > 0) & (gamma < 2)
        valid_transition_time = (0.25 * self.max_time < transition_time) & (
            transition_time < 0.9 * self.max_time
        )
        valid_params_mask = (
            valid_log10_init_energy & valid_gamma & valid_transition_time
        )
        if num_local_walkers == 1:
            return valid_params_mask[0]
        return valid_params_mask

    @override
    def _annotate_fitted_params(self, axs: list["Axes"]) -> None:
        log10_gamma_samples = self.log10_e * self.fitted_posterior_samples[:, 1]
        transition_time_samples = self.fitted_posterior_samples[:, 2]
        plot_param_percentiles(axs[1], log10_gamma_samples, orientation="horizontal")
        for row_index in range(len(axs)):
            plot_param_percentiles(
                axs[row_index], transition_time_samples, orientation="vertical"
            )

    @override
    def _get_output_params(self) -> tuple["npt.NDArray[np.float64]", list[str]]:
        log10_init_energy_samples = self.fitted_posterior_samples[:, 0]
        gamma_samples = self.fitted_posterior_samples[:, 1]
        transition_time_samples = self.fitted_posterior_samples[:, 2]
        log10_sat_energy_samples = (
            log10_init_energy_samples
            + self.log10_e * gamma_samples * transition_time_samples
        )
        output_param_samples = np.column_stack([
            log10_init_energy_samples,
            log10_sat_energy_samples,
            gamma_samples,
        ])
        output_param_labels = [
            r"$\log_{10}(E_{\mathrm{init}})$",
            r"$\log_{10}(E_{\mathrm{sat}})$",
            r"$\gamma$",
        ]
        return output_param_samples, output_param_labels

    @override
    def _annotate_output_params(self, axs: list["Axes"]) -> None:
        log10_sat_energy_samples = self.output_posterior_samples[:, 1]
        plot_param_percentiles(
            axs[0], log10_sat_energy_samples, orientation="horizontal"
        )


# END OF MODULE
