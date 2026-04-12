## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from collections import deque
from abc import ABC, abstractmethod
from typing import Any, Callable

## third-party
import numpy
import emcee
from tqdm import tqdm
from scipy.stats import gaussian_kde
from numpy.typing import NDArray

## personal
from jormi.ww_io import manage_io

## local
from . import plot_chain_evolution
from . import plot_model_posteriors
from . import plot_model_fits

##
## === BASE ROUTINE
##


class BaseMCMCRoutine(ABC):
    """
  Base class for running MCMC inference using emcee. 
  Subclasses must define a model and parameter validation logic.
  """

    ## methods that need to be implemented by each subclass

    @abstractmethod
    def _model(
        self,
        param_vectors: NDArray[Any],
    ) -> NDArray[Any]: ...

    @abstractmethod
    def _get_valid_params_mask(
        self,
        param_vectors: NDArray[Any],
    ) -> NDArray[Any]: ...

    ## hooks that can be overwritten by each subclass

    def _get_kde_params(
        self,
        param_vectors: NDArray[Any],
    ) -> NDArray[Any]:
        return numpy.asarray(param_vectors)

    def _get_output_params(
        self,
    ) -> tuple[NDArray[Any], list[str]]:
        assert self.fitted_posterior_samples is not None
        return self.fitted_posterior_samples, self.fitted_param_labels

    def _annotate_fitted_params(
        self,
        _axs: Any,
    ) -> None:
        pass

    def _annotate_output_params(
        self,
        _axs: Any,
    ) -> None:
        pass

    ## core functionality

    def __init__(
        self,
        *,
        routine_name: str,
        output_directory: str | Path,
        x_values: list[Any] | NDArray[Any],
        y_values: list[Any] | NDArray[Any],
        likelihood_sigma: list[Any] | NDArray[Any],
        initial_params: tuple[float, ...],
        prior_kde: gaussian_kde | None = None,
        plot_posterior_kde: bool = False,
        data_label: str | None = None,
        fitted_param_labels: list[str] | None = None,
    ) -> None:
        self.routine_name = routine_name
        self.output_directory = output_directory
        self.x_values = numpy.asarray(x_values)
        self.y_values = numpy.asarray(y_values)
        self.likelihood_sigma = likelihood_sigma
        self.initial_params = initial_params
        self.num_params = len(self.initial_params)
        self._prior_logpdf: Callable[..., Any] | None = prior_kde.logpdf if (prior_kde is not None) else None
        self.data_label = data_label
        self.fitted_param_labels = fitted_param_labels if (fitted_param_labels is not None) else []
        self.plot_posterior_kde = plot_posterior_kde
        self._validate_inputs()
        ## define key outputs
        self.raw_chain: NDArray[Any] | None = None
        self.auto_correlation_time: NDArray[Any] | None = None
        self.fitted_posterior_samples: NDArray[Any] | None = None
        self.fitted_posterior_kde: gaussian_kde | None = None
        self.output_posterior_samples: NDArray[Any] | None = None
        self.output_posterior_kde: gaussian_kde | None = None

    def _validate_inputs(
        self,
    ) -> None:
        if not isinstance(self.x_values, (list, numpy.ndarray)):
            raise ValueError(f"`x_values` should either be a list or array of values.")
        if not isinstance(self.y_values, (list, numpy.ndarray)):
            raise ValueError(f"`y_values` should either be a list or array of values.")
        if not isinstance(self.likelihood_sigma, (list, numpy.ndarray)):
            raise ValueError(f"`likelihood_sigma` should either be a list or array of values.")
        if len(self.x_values) != len(self.y_values):
            raise ValueError(
                f"`x_values` and `y_values` should be the same length. Received {len(self.x_values)} vs {len(self.y_values)}.",
            )
        if len(self.x_values) != len(self.likelihood_sigma):
            raise ValueError(
                f"`x_values` and `likelihood_sigma` should be the same length. Received {len(self.x_values)} vs {len(self.likelihood_sigma)}.",
            )
        if not isinstance(self.initial_params, tuple):
            raise ValueError(
                f"`initial_params` should be a tuple. Received {type(self.initial_params)}.",
            )

    def estimate_posterior(
        self,
        num_walkers_per_param: int = 10,
        num_steps: int = 10_000,
        burn_in_steps: int = 3_000,
    ) -> None:
        if not numpy.all(self._get_valid_params_mask(numpy.asarray(self.initial_params))):
            raise ValueError(f"Initial guess is invalid!")
        print("Estimating the posterior...")
        num_params = len(self.initial_params)
        self.num_walkers = num_walkers_per_param * num_params
        self.num_steps = num_steps
        perturbed_params = numpy.array(
            self.initial_params,
        ) + 1e-3 * numpy.random.randn(
            self.num_walkers,
            self.num_params,
        )
        mcmc_sampler = emcee.EnsembleSampler(
            nwalkers=self.num_walkers,
            ndim=self.num_params,
            log_prob_fn=self._log_posterior,
            vectorize=True,
        )
        deque(
            tqdm(
                mcmc_sampler.sample(initial_state=perturbed_params, iterations=int(num_steps)),
                total=int(num_steps),
            ),
            maxlen=0,  # discard returned samples; deque is only used to force evaluation of the generator
        )
        ## save key outputs
        acc_fracs = mcmc_sampler.acceptance_fraction  # shape: (nwalkers,)
        median_acc = float(numpy.median(acc_fracs))
        low, high = float(numpy.min(acc_fracs)), float(numpy.max(acc_fracs))
        print(f"Acceptance fraction: median={median_acc:.3f} [{low:.3f}, {high:.3f}]")
        self.raw_chain = mcmc_sampler.get_chain()
        self._check_chain_convergence(mcmc_sampler)
        self.fitted_posterior_samples = mcmc_sampler.get_chain(
            discard=int(burn_in_steps),
            # thin=10,
            flat=True,
        )
        assert self.fitted_posterior_samples is not None
        self.fitted_log_likelihoods = self._log_likelihood(self.fitted_posterior_samples)
        self.output_posterior_samples, self.output_param_labels = self._get_output_params()
        assert self.output_posterior_samples is not None
        if numpy.array_equal(self.output_posterior_samples, self.fitted_posterior_samples):
            print("Estimating the KDE of only the fitted posterior...")
            self.fitted_posterior_kde = gaussian_kde(
                self.fitted_posterior_samples.T,
                bw_method="scott",
            )
            self.output_posterior_kde = self.fitted_posterior_kde
        else:
            print("Estimating the KDE of both the fitted and output posteriors...")
            self.fitted_posterior_kde = gaussian_kde(
                self.fitted_posterior_samples.T,
                bw_method="scott",
            )
            self.output_posterior_kde = gaussian_kde(
                self.output_posterior_samples.T,
                bw_method="scott",
            )
        ## create diagnostic outputs
        self._make_plots()
        self._save_posterior_samples()

    def _log_posterior(
        self,
        param_vectors: NDArray[Any],
    ) -> NDArray[Any]:
        lp_values = self._log_prior(param_vectors)
        valid_prior_mask = numpy.isfinite(lp_values)
        ll_values = numpy.full_like(lp_values, -numpy.inf)
        ll_values[valid_prior_mask] = self._log_likelihood(param_vectors[valid_prior_mask])
        return lp_values + ll_values

    def _log_prior(
        self,
        param_vectors: NDArray[Any],
    ) -> NDArray[Any]:
        valid_params_mask = self._get_valid_params_mask(param_vectors)
        num_local_walkers = param_vectors.shape[0]
        lp_values = numpy.full(num_local_walkers, -numpy.inf)
        if self._prior_logpdf is not None:
            valid_params = numpy.atleast_2d(param_vectors[valid_params_mask])
            kde_vector = self._get_kde_params(valid_params)
            kde_logpdfs = numpy.asarray(self._prior_logpdf(kde_vector.T))
            lp_values[valid_params_mask] = kde_logpdfs
        else:
            lp_values[valid_params_mask] = 0.0  # uniform prior
        return lp_values

    def _log_likelihood(
        self,
        param_vectors: NDArray[Any],
    ) -> NDArray[Any]:
        param_vectors = numpy.atleast_2d(param_vectors)
        num_local_walkers = param_vectors.shape[0]
        valid_params_mask = self._get_valid_params_mask(param_vectors)
        ll_values = numpy.full(num_local_walkers, -numpy.inf)
        if not numpy.any(valid_params_mask):
            return ll_values
        try:
            valid_params = param_vectors[valid_params_mask]
            modelled_y = numpy.atleast_2d(self._model(valid_params))
            measured_y = numpy.asarray(self.y_values)
            assert modelled_y.shape == (valid_params.shape[0], measured_y.shape[0]), (
                f"Expected model output shape ({valid_params.shape[0]}, {measured_y.shape[0]}), got {modelled_y.shape}"
            )
            ## add a leading dimension to the measured data so it broadcasts across the vectorised param-rows
            y_residuals_2d = measured_y[None, :] - modelled_y
            likelihood_sigma_2d = numpy.asarray(self.likelihood_sigma)[None, :]
            ll_values[valid_params_mask] = -0.5 * numpy.sum(
                numpy.square(y_residuals_2d / likelihood_sigma_2d),
                axis=1,
            )
        except Exception as error:
            raise
        return ll_values

    def _check_chain_convergence(self, mcmc_sampler: emcee.EnsembleSampler) -> dict[str, Any]:
        """
        Run lightweight convergence diagnostics on the *raw* emcee chain.
        Reports (a) acceptance fractions, (b) integrated autocorrelation times tau,
        and (c) approximate effective sample sizes (ESS). Sets attributes on `self`
        and returns a summary dict.
        """
        summary: dict[str, Any] = {}
        # --- Acceptance fraction diagnostics (always available)
        acc_fracs = numpy.asarray(mcmc_sampler.acceptance_fraction, dtype=float)
        if acc_fracs.size:
            acc_med = float(numpy.median(acc_fracs))
            acc_min = float(numpy.min(acc_fracs))
            acc_max = float(numpy.max(acc_fracs))
            summary["acceptance_fraction"] = {
                "median": acc_med,
                "min": acc_min,
                "max": acc_max,
                "per_walker": acc_fracs.tolist(),
            }
            # Heuristic guidance band
            if acc_med < 0.15:
                print(f"WARNING: Low median acceptance fraction ({acc_med:.3f}); chains may be stuck.")
            elif acc_med > 0.70:
                print(f"WARNING: High median acceptance fraction ({acc_med:.3f}); steps may be too small.")
            else:
                print(f"Acceptance fraction OK: median={acc_med:.3f} [{acc_min:.3f}, {acc_max:.3f}]")
            # Store on self for later reporting if you like
            self.acceptance_fraction = acc_fracs
        else:
            print("NOTE: Sampler did not report acceptance fractions.")
        # --- Autocorrelation time and ESS
        nwalk = self.num_walkers
        nstep = int(self.num_steps)
        try:
            # tol=0 gives the most conservative estimate; it will raise if chain is too short
            tau = numpy.asarray(mcmc_sampler.get_autocorr_time(tol=0), dtype=float)  # shape: (ndim,)
            self.auto_correlation_time = tau
            tau_med = float(numpy.median(tau))
            tau_max = float(numpy.max(tau))
            summary["autocorr_time"] = {"per_param": tau.tolist(), "median": tau_med, "max": tau_max}
            # Effective sample size per parameter (rough heuristic): N / (2*tau)
            # where N = nwalkers * nsteps (raw, unthinned)
            N_raw = nwalk * nstep
            ess = (N_raw / (2.0 * tau))
            ess_med = float(numpy.median(ess))
            ess_min = float(numpy.min(ess))
            summary["ess"] = {
                "per_param": ess.tolist(),
                "median": ess_med,
                "min": ess_min,
                "N_raw": int(N_raw),
            }
            # Simple convergence gates based on tau
            # - Good: nstep >= 50 * tau_max
            # - Borderline: 5 * tau_max <= nstep < 50 * tau_max
            # - Poor: nstep < 5 * tau_max
            if nstep >= 50 * tau_max:
                print(f"Chains appear converged: n_steps={nstep} >= 50x tau_max~{tau_max:.1f}.")
            elif nstep >= 5 * tau_max:
                print(
                    f"WARNING: Borderline length: n_steps={nstep} is between 5x and 50x tau_max~{tau_max:.1f}.",
                )
            else:
                print(f"WARNING: Chain likely too short: n_steps={nstep} < 5x tau_max~{tau_max:.1f}.")
            print(
                f"Autocorr time (median/max): {tau_med:.1f}/{tau_max:.1f} steps; "
                f"approx. ESS median/min: {ess_med:.0f}/{ess_min:.0f} out of N={N_raw} raw samples.",
            )
        except emcee.autocorr.AutocorrError:
            # Fall back to a softer message if tau is unreliable at current length
            print(
                "WARNING: Could not reliably estimate autocorrelation time (chain likely too short). "
                "Proceeding with visual/heuristic checks.",
            )
            self.auto_correlation_time = None
            summary["autocorr_time"] = None
            summary["ess"] = None
        # Optionally store a compact dict on self for later reporting
        self.convergence_summary = summary
        return summary

    def _make_plots(
        self,
    ) -> None:
        plot_chain_evolution.PlotChainEvolution(self).plot()
        plot_model_posteriors.PlotModelPosteriors(self).plot()
        plot_model_fits.PlotModelFits(self).plot()

    def _save_posterior_samples(
        self,
    ) -> None:
        fitted_posterior_path = manage_io.combine_file_path_parts(
            [self.output_directory, f"{self.routine_name}_fitted_posterior_samples.npy"],
        )
        output_posterior_path = manage_io.combine_file_path_parts(
            [self.output_directory, f"{self.routine_name}_output_posterior_samples.npy"],
        )
        log_likelihood_path = manage_io.combine_file_path_parts(
            [self.output_directory, f"{self.routine_name}_fitted_log_likelihoods.npy"],
        )
        assert self.fitted_posterior_samples is not None
        assert self.output_posterior_samples is not None
        numpy.save(fitted_posterior_path, self.fitted_posterior_samples)
        numpy.save(log_likelihood_path, self.fitted_log_likelihoods)
        if not numpy.array_equal(self.output_posterior_samples, self.fitted_posterior_samples):
            numpy.save(output_posterior_path, self.output_posterior_samples)


## } MODULE
