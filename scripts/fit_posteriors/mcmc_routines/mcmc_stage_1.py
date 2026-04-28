## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray

## local
from . import mcmc_base
from . import mcmc_utils

##
## === STAGE 1 MCMC FITTER
##


class Stage1MCMCRoutine(mcmc_base.BaseMCMCRoutine):

    def __init__(
        self,
        *,
        output_directory: str | Path,
        time_values: list[Any] | NDArray[Any],
        ave_log10_energy_values: list[Any] | NDArray[Any],
        std_log10_energy_values: list[Any] | NDArray[Any],
        initial_params: tuple[float, ...],
        plot_posterior_kde: bool = True,
    ) -> None:
        assert len(
            initial_params,
        ) == 3, ("Stage 1 MCMC routine expects 3 initial params: log10(E_init), gamma_exp, and t_approx")
        self.log10_e: float = numpy.log10(
            numpy.exp(
                1,
            ),
        )
        self.max_sim_time: float = numpy.max(time_values)
        super().__init__(
            routine_name="stage1",
            output_directory=output_directory,
            x_values=time_values,
            y_values=ave_log10_energy_values,
            likelihood_sigma=std_log10_energy_values,
            initial_params=initial_params,
            plot_posterior_kde=plot_posterior_kde,
            data_label=r"$\log_{10}(E_{\mathrm{mag}})$",
            fitted_param_labels=[
                r"$\log_{10}(E_{\mathrm{init}})$",
                r"$\gamma$",
                r"$t_{\mathrm{approx}}$",
            ],
        )

    def _model(
        self,
        param_vectors: NDArray[Any],
    ) -> NDArray[Any]:
        param_vectors = numpy.atleast_2d(param_vectors)  # (N, P)
        ## output dimensions
        num_local_walkers = param_vectors.shape[0]  # N
        num_data_points = len(self.x_values)  # T
        ## unpack model parameters (P = 3)
        log10_init_energy = param_vectors[:, 0]
        gamma = param_vectors[:, 1]
        transition_time = param_vectors[:, 2]
        ## reshape parameters to allow for vectorising over param-rows
        x_values_2d = self.x_values[None, :]  # shape (1, T)
        gamma_2d = gamma[:, None]  # shape (N, 1)
        transition_time_2d = transition_time[:, None]  # shape (N, 1)
        log10_init_energy_2d = log10_init_energy[:, None]  # shape (N, 1)
        ## mask (reduced) SSD phases
        mask_exp_phase = x_values_2d < transition_time_2d
        mask_sat_phase = ~mask_exp_phase
        ## compute (reduced) SSD phases
        log10_energy_exp_values = log10_init_energy_2d + self.log10_e * gamma_2d * x_values_2d  # (N, T)
        log10_energy_sat_values = log10_init_energy_2d + self.log10_e * gamma_2d * transition_time_2d  # (N, 1)
        log10_energy_sat_values = numpy.broadcast_to(
            log10_energy_sat_values,
            (num_local_walkers, num_data_points),
        )  # (N, T)
        ## assemble modelled (reduced) SSD phases
        log10_energy = numpy.empty((num_local_walkers, num_data_points))
        log10_energy[mask_exp_phase] = log10_energy_exp_values[mask_exp_phase]
        log10_energy[mask_sat_phase] = log10_energy_sat_values[mask_sat_phase]
        return log10_energy

    def _get_valid_params_mask(
        self,
        param_vectors: NDArray[Any],
        verbose: bool = False,
    ) -> NDArray[Any]:
        param_vectors = numpy.atleast_2d(param_vectors)
        num_local_walkers = param_vectors.shape[0]
        log10_init_energy = param_vectors[:, 0]
        gamma = param_vectors[:, 1]
        transition_time = param_vectors[:, 2]
        valid_log10_init_energy = (-30 < log10_init_energy) & (log10_init_energy < -5)
        valid_gamma = (0 < gamma) & (gamma < 10)
        valid_transition_time = (0.1 * self.max_sim_time
                                 < transition_time) & (transition_time < 0.9 * self.max_sim_time)
        valid_params_mask = valid_log10_init_energy & valid_gamma & valid_transition_time
        if verbose and not numpy.all(valid_params_mask):
            checks = [
                ("log10_init_energy", valid_log10_init_energy),
                ("gamma_exp", valid_gamma),
                ("t_approx", valid_transition_time),
            ]
            invalid_params = [
                (param_name, param_valid_mask)
                for param_name, param_valid_mask in checks
                if not numpy.all(param_valid_mask)
            ]
            message_parts = [
                f"{param_name} ({numpy.count_nonzero(~param_valid_mask)}/{num_local_walkers})"
                for param_name, param_valid_mask in invalid_params
            ]
            print(f"[Stage1] invalid parameters: {', '.join(message_parts)}")
        if num_local_walkers == 1:
            return valid_params_mask[0]
        return valid_params_mask

    def _annotate_fitted_params(
        self,
        axs: Any,
    ) -> None:
        assert self.fitted_posterior_samples is not None
        log10_gamma_samples = self.log10_e * self.fitted_posterior_samples[:, 1]
        transition_time_samples = self.fitted_posterior_samples[:, 2]
        mcmc_utils.plot_param_percentiles_h(axs[1], log10_gamma_samples)
        for row_index in range(len(axs)):
            mcmc_utils.plot_param_percentiles_v(axs[row_index], transition_time_samples)

    def _get_output_params(
        self,
    ) -> tuple[NDArray[Any], list[str]]:
        assert self.fitted_posterior_samples is not None
        log10_init_energy_samples = self.fitted_posterior_samples[:, 0]
        gamma_samples = self.fitted_posterior_samples[:, 1]
        transition_time_samples = self.fitted_posterior_samples[:, 2]
        log10_sat_energy_samples = log10_init_energy_samples + self.log10_e * gamma_samples * transition_time_samples
        output_param_samples = numpy.column_stack(
            [
                log10_init_energy_samples,
                log10_sat_energy_samples,
                gamma_samples,
            ],
        )
        output_param_labels = [
            r"$\log_{10}(E_{\mathrm{init}})$",
            r"$\log_{10}(E_{\mathrm{sat}})$",
            r"$\gamma$",
        ]
        return output_param_samples, output_param_labels

    def _annotate_output_params(
        self,
        axs: Any,
    ) -> None:
        assert self.output_posterior_samples is not None
        log10_sat_energy_samples = self.output_posterior_samples[:, 1]
        mcmc_utils.plot_param_percentiles_h(axs[0], log10_sat_energy_samples)


## } MODULE
