## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
import numpy
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from numpy.typing import NDArray

## personal
from jormi import ww_lists

## local
from . import mcmc_base
from . import mcmc_utils

##
## === UNIFIED STAGE 2 MCMC FITTER
##


class Stage2MCMCRoutine(
        mcmc_base.BaseMCMCRoutine, ):

    def __init__(
        self,
        *,
        output_directory: str,
        time_values: list[Any] | NDArray[Any],
        ave_energy_values: list[Any] | NDArray[Any],
        std_energy_values: list[Any] | NDArray[Any],
        initial_params: tuple[float, float, float, float],
        prior_kde: gaussian_kde | None = None,
        plot_posterior_kde: bool = True,
        fixed_nl_exponent: float | None = None,
        routine_name: str = "stage2",
    ) -> None:
        assert len(initial_params) == 4, (
            "Stage 2 MCMC routine expects 4 initial params: log10(E_init), log10(E_sat), gamma_exp, and t_nl"
        )
        guess_sat_time = self._define_constraints(
            time_values=time_values,
            ave_energy_values=ave_energy_values,
        )
        fitted_param_labels = [
            r"$\log_{10}(E_{\mathrm{init}})$",
            r"$\log_{10}(E_{\mathrm{sat}})$",
            r"$\gamma_\mathrm{exp}$",
            r"$t_{\mathrm{nl}}$",
            r"$t_{\mathrm{sat}}$",
        ]
        all_initial_params = [
            initial_params[0],  # log10(E_init)
            initial_params[1],  # log10(E_sat)
            initial_params[2],  # exp_gamma
            initial_params[3],  # t_nl
            guess_sat_time,  # t_sat
        ]
        self.fixed_nl_exponent: float | None = fixed_nl_exponent
        if self.fixed_nl_exponent is None:
            fitted_param_labels.append(r"$p$")
            all_initial_params.append(1.5)  # nl_exponent
        else:
            assert 1.0 <= self.fixed_nl_exponent <= 2.0, "provided `fixed_nl_exponent` should be in [1, 2]"
        super().__init__(
            routine_name=routine_name,
            output_directory=output_directory,
            x_values=time_values,
            y_values=ave_energy_values,
            likelihood_sigma=std_energy_values,
            initial_params=tuple(all_initial_params),
            prior_kde=prior_kde,
            plot_posterior_kde=plot_posterior_kde,
            data_label=r"$E_{\mathrm{mag}}$",
            fitted_param_labels=fitted_param_labels,
        )

    def _define_constraints(
        self,
        time_values: list[Any] | NDArray[Any],
        ave_energy_values: list[Any] | NDArray[Any],
        max_num_bins: int = 100,
    ) -> float:
        self.max_sim_time: float = float(numpy.max(time_values))
        if len(time_values) > max_num_bins:
            time_bin_edges = numpy.linspace(
                numpy.min(time_values),
                numpy.max(time_values),
                max_num_bins + 1,
            )
            time_bin_indices = numpy.digitize(time_values, time_bin_edges) - 1
            binned_time_values = []
            binned_ave_energy_values = []
            for time_bin_index in range(max_num_bins):
                time_bin_mask = (time_bin_indices == time_bin_index)
                if not numpy.any(time_bin_mask):
                    continue
                binned_time_values.append(numpy.mean(time_values[time_bin_mask]))
                binned_ave_energy_values.append(numpy.mean(ave_energy_values[time_bin_mask]))
            used_time_values = numpy.asarray(binned_time_values)
            used_ave_energy_values = numpy.asarray(binned_ave_energy_values)
        else:
            used_time_values = numpy.asarray(time_values)
            used_ave_energy_values = numpy.asarray(ave_energy_values)
        ## define max time to transition into saturated phase
        dlny_dt = gaussian_filter1d(
            numpy.gradient(
                numpy.log10(used_ave_energy_values),
                used_time_values,
            ),
            sigma=2,
        )
        max_sat_time_index = ww_lists.get_index_of_first_crossing(
            values=[float(value) for value in dlny_dt],
            target=0,
        )
        self.max_sat_time: float = float(used_time_values[max_sat_time_index])
        ## define max time to transition into non-linear phase
        ## note, make sure this happens before the saturated phase
        dy_dt = numpy.gradient(
            used_ave_energy_values,
            used_time_values,
        )
        ## use half the peak dE/dt before saturation as the threshold for the non-linear transition
        target_dy_dt = float(0.5 * numpy.max(dy_dt[:max_sat_time_index]))
        max_nl_time_index = ww_lists.get_index_of_first_crossing(
            values=[float(value) for value in dy_dt[:max_sat_time_index]],
            target=target_dy_dt,
        )
        self.max_nl_time: float = float(used_time_values[max_nl_time_index])
        ## construct a valid guess for the transition time into the saturated phase
        guess_sat_time = self.max_nl_time + 0.5 * (self.max_sat_time - self.max_nl_time)
        return float(guess_sat_time)

    def _model(
        self,
        param_vectors: NDArray[Any],
    ) -> NDArray[Any]:
        param_vectors = numpy.atleast_2d(param_vectors)  # (N, P)
        ## output dimensions
        num_local_walkers = param_vectors.shape[0]  # N
        num_data_points = len(self.x_values)  # T
        if self.fixed_nl_exponent is None:
            ## unpack model parameters (P = 6)
            log10_init_energy = param_vectors[:, 0]
            log10_sat_energy = param_vectors[:, 1]
            exp_gamma = param_vectors[:, 2]
            nl_start_time = param_vectors[:, 3]
            sat_start_time = param_vectors[:, 4]
            nl_exponent = param_vectors[:, 5]
        else:
            ## unpack model parameters (P = 5)
            log10_init_energy = param_vectors[:, 0]
            log10_sat_energy = param_vectors[:, 1]
            exp_gamma = param_vectors[:, 2]
            nl_start_time = param_vectors[:, 3]
            sat_start_time = param_vectors[:, 4]
            nl_exponent = self.fixed_nl_exponent * numpy.ones_like(log10_init_energy)
        ## reshape parameters to allow for vectorising over param-rows
        x_values_2d = self.x_values[None, :]  # shape (1, T)
        exp_gamma_2d = exp_gamma[:, None]  # shape (N, 1)
        nl_start_time_2d = nl_start_time[:, None]  # shape (N, 1)
        sat_start_time_2d = sat_start_time[:, None]  # shape (N, 1)
        nl_exponent_2d = nl_exponent[:, None]  # shape (N, 1)
        ## mask SSD phases
        mask_exp_phase = x_values_2d < nl_start_time_2d
        mask_nl_phase = (nl_start_time_2d <= x_values_2d) & (x_values_2d < sat_start_time_2d)
        mask_sat_phase = sat_start_time_2d < x_values_2d
        ## compute model constants (per walker)
        init_energy = 10**log10_init_energy  # (N,)
        init_energy_2d = init_energy[:, None]  # (N, 1)
        sat_energy = 10**log10_sat_energy  # (N,)
        sat_energy_2d = sat_energy[:, None]  # (N, 1)
        nl_start_energy = init_energy * numpy.exp(exp_gamma * nl_start_time)  # (N,)
        nl_start_energy_2d = nl_start_energy[:, None]  # (N, 1)
        nl_gamma = (sat_energy - nl_start_energy) / (sat_start_time - nl_start_time)**nl_exponent  # (N,)
        nl_gamma_2d = nl_gamma[:, None]  # (N, 1)
        ## assemble modelled SSD phases
        energy_2d = numpy.zeros((num_local_walkers, num_data_points))
        energy_2d[mask_exp_phase] = (init_energy_2d * numpy.exp(exp_gamma_2d * x_values_2d))[mask_exp_phase
                                                                                             ]  # (N, T)
        energy_2d[mask_nl_phase] = (
            nl_start_energy_2d + nl_gamma_2d * (x_values_2d - nl_start_time_2d)**nl_exponent_2d
        )[mask_nl_phase]  # (N, T)
        energy_2d[mask_sat_phase] = numpy.broadcast_to(
            sat_energy_2d,
            (num_local_walkers, num_data_points),
        )[mask_sat_phase]  # (N, T)
        return energy_2d

    def _get_valid_params_mask(
        self,
        param_vectors: NDArray[Any],
        verbose: bool = False,
    ) -> NDArray[Any]:
        param_vectors = numpy.atleast_2d(param_vectors)
        num_local_walkers = param_vectors.shape[0]
        if self.fixed_nl_exponent is None:
            log10_init_energy = param_vectors[:, 0]
            log10_sat_energy = param_vectors[:, 1]
            exp_gamma = param_vectors[:, 2]
            nl_start_time = param_vectors[:, 3]
            sat_start_time = param_vectors[:, 4]
            nl_exponent = param_vectors[:, 5]
        else:
            log10_init_energy = param_vectors[:, 0]
            log10_sat_energy = param_vectors[:, 1]
            exp_gamma = param_vectors[:, 2]
            nl_start_time = param_vectors[:, 3]
            sat_start_time = param_vectors[:, 4]
            nl_exponent = self.fixed_nl_exponent * numpy.ones_like(log10_init_energy)
        valid_log10_init_energy = (-30 < log10_init_energy) & (log10_init_energy < -5)
        valid_log10_sat_energy = (-5 < log10_sat_energy) & (log10_sat_energy < 0)
        valid_exp_gamma = (0 < exp_gamma) & (exp_gamma < 10)
        valid_nl_start_time = (nl_start_time < self.max_nl_time) & (nl_start_time < sat_start_time)
        valid_sat_start_time = sat_start_time < self.max_sat_time
        valid_nl_exponent = (1.0 <= nl_exponent) & (nl_exponent <= 2.0)
        valid_params_mask = (
            valid_log10_init_energy & valid_log10_sat_energy & valid_exp_gamma & valid_nl_start_time
            & valid_sat_start_time & valid_nl_exponent
        )
        if verbose and not numpy.all(valid_params_mask):
            checks = [
                ("log10_init_energy", valid_log10_init_energy),
                ("log10_sat_energy", valid_log10_sat_energy),
                ("exp_gamma", valid_exp_gamma),
                ("nl_start_time", valid_nl_start_time),
                ("sat_start_time", valid_sat_start_time),
                ("nl_exponent", valid_nl_exponent),
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
            print(f"[Stage2] invalid parameters: {', '.join(message_parts)}")
        if num_local_walkers == 1:
            return valid_params_mask[0]
        return valid_params_mask

    def _get_kde_params(
        self,
        param_vectors: NDArray[Any],
    ) -> NDArray[Any]:
        param_vectors = numpy.atleast_2d(param_vectors)
        ## ignore the transition times implicity gives them a unifrom prior
        return numpy.asarray(param_vectors[:, :3])

    def _annotate_fitted_params(
        self,
        axs: Any,
    ) -> None:
        assert self.fitted_posterior_samples is not None
        sat_energy_samples = 10**self.fitted_posterior_samples[:, 1]
        nl_start_time_samples = self.fitted_posterior_samples[:, 3]
        sat_start_time_samples = self.fitted_posterior_samples[:, 4]
        mcmc_utils.plot_param_percentiles_h(axs[0], sat_energy_samples)
        for row_index in range(len(axs)):
            mcmc_utils.plot_param_percentiles_v(axs[row_index], nl_start_time_samples)
            mcmc_utils.plot_param_percentiles_v(axs[row_index], sat_start_time_samples)
            axs[row_index].axvline(
                self.max_nl_time,
                color="red",
                ls="--",
            )
            axs[row_index].axvline(
                self.max_sat_time,
                color="red",
                ls="--",
            )


##
## === WRAPPERS EXPOSING MODEL VARIANTS
##


class Stage2MCMCRoutine_free(
        Stage2MCMCRoutine, ):

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fixed_nl_exponent=None,
            routine_name="stage2_free",
            **kwargs,
        )


class Stage2MCMCRoutine_linear(
        Stage2MCMCRoutine, ):

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fixed_nl_exponent=1.0,
            routine_name="stage2_linear",
            **kwargs,
        )


class Stage2MCMCRoutine_quadratic(
        Stage2MCMCRoutine, ):

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fixed_nl_exponent=2.0,
            routine_name="stage2_quadratic",
            **kwargs,
        )


## } MODULE
