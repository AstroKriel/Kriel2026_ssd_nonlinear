# START OF MODULE


# ###############################################################
# DEPENDENCIES
# ###############################################################

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from pathlib import Path
from collections import deque

from tqdm import tqdm
import emcee
import numpy as np
from jormi.ww_io import io_manager
from scipy.stats import gaussian_kde

from . import plot_model_fits, plot_chain_evolution, plot_model_posteriors

if TYPE_CHECKING:
    from types import FunctionType

    from numpy import typing as npt
    from matplotlib.axes import Axes

# ###############################################################
# BASE ROUTINE
# ###############################################################


class BaseMCMCRoutine(ABC):
    """Base class for running MCMC inference using emcee.

    Subclasses must define a model and parameter validation logic.
    """

    # core functionality
    # SUGGESTION: Always put __init__ first, unless the things above are classmethods or classvars
    def __init__(
        self,
        *,
        routine_name: str,
        output_directory: "str | Path",
        x_values: "list[float] | npt.NDArray[np.float64]",
        y_values: "list[float] | npt.NDArray[np.float64]",
        initial_params: tuple[float, ...],
        prior_kde: "gaussian_kde | None" = None,
        likelihood_sigma: float = 1.0,
        plot_posterior_kde: bool = False,
        data_label: str | None = None,
        fitted_param_labels: list[str] | None = None,
    ) -> None:
        if fitted_param_labels is None:
            fitted_param_labels = []
        self.routine_name: str = routine_name
        self.output_directory: Path = Path(output_directory)
        self.x_values: npt.NDArray[np.float64] = np.asarray(x_values, dtype=np.float64)
        self.y_values: npt.NDArray[np.float64] = np.asarray(y_values, dtype=np.float64)
        self.initial_params: tuple[float, ...] = initial_params
        self.num_params: int = len(self.initial_params)
        self._prior_logpdf: FunctionType | None = (
            prior_kde.logpdf if prior_kde is not None else None
        )
        self.likelihood_sigma: float = likelihood_sigma
        self.data_label: str | None = data_label
        self.fitted_param_labels: list[str] = fitted_param_labels
        self.plot_posterior_kde: bool = plot_posterior_kde
        self._validate_inputs()  # TODO(AstroKriel): Why is this necessary? If this is because the user can pass incorrect data in, then that should be handled when the user first passes data in.

        # SUGGESTION: Define all class variables here, even if they're instantiated later
        # mcmc variables
        self.num_walkers: int
        self.num_steps: int

        # define key outputs
        # SUGGESTION: You can define variables without instantiating them
        self.raw_chain: npt.NDArray[np.float64]
        self.auto_correlation_time: npt.NDArray[np.float64]
        self.fitted_posterior_samples: npt.NDArray[np.float64]
        self.fitted_posterior_kde: gaussian_kde
        self.output_posterior_samples: npt.NDArray[np.float64]
        self.output_posterior_kde: gaussian_kde
        self.output_param_labels: list[str]

    # TODO(AstroKriel): See if you can refactor so that this isn't necessary
    def _validate_inputs(self) -> None:
        if not isinstance(self.x_values, (list, np.ndarray)):
            msg = "`x_values` should either be a list or array of values."
            raise ValueError(msg)
        if not isinstance(self.y_values, (list, np.ndarray)):
            msg = "`y_values` should either be a list or array of values."
            raise ValueError(msg)
        if len(self.x_values) != len(self.y_values):
            msg = f"`x_values` and `y_values` should be the same length. Received {len(self.x_values)} vs {len(self.y_values)}."
            raise ValueError(msg)
        if not isinstance(self.initial_params, tuple):
            msg = f"`initial_params` should be a tuple. Received {type(self.initial_params)}."
            raise ValueError(msg)
        if not isinstance(self.likelihood_sigma, float):
            msg = "`likelihood_sigma` should be a scalar."
            raise ValueError(msg)

    # methods that need to be implemented by each subclass
    # SUGGESTION: The [ABC](https://docs.python.org/3/library/abc.html) python library allows you to enforce this
    @abstractmethod
    def _model(
        self, param_vectors: "tuple[float, ...] | npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.float64]": ...

    @abstractmethod
    def _get_valid_params_mask(
        self, param_vectors: "tuple[float, ...] | npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.bool]": ...

    @abstractmethod
    def _annotate_fitted_params(self, _axs: list["Axes"]) -> None: ...

    @abstractmethod
    def _annotate_output_params(self, _axs: list["Axes"]) -> None: ...

    # hooks that can be overwritten by each subclass
    @staticmethod  # TODO(AstroKriel): if you don't use self or cls, make it a static method
    def _get_kde_params(
        param_vectors: "tuple[float, ...] | npt.NDArray[np.float64]",
    ) -> "npt.NDArray[np.float64]":
        return np.asarray(param_vectors)

    def _get_output_params(self) -> tuple["npt.NDArray[np.float64]", list[str]]:
        return self.fitted_posterior_samples, self.fitted_param_labels

    def estimate_posterior(
        self,
        num_walkers: int = 200,
        num_steps: int = 7000,
        burn_in_steps: int = 1000,
    ) -> None:
        if not self._get_valid_params_mask(self.initial_params):
            msg = "Initial guess is invalid!"
            raise ValueError(msg)
        print(
            "Estimating the posterior..."
        )  # TODO(AstroKriel): Use logging. See [here](https://github.com/OmegaLambda1998/suPAErnova/blob/main/src/suPAErnova/logging.py) for an example.
        self.num_walkers = num_walkers
        self.num_steps = num_steps
        perturbed_params = (
            np.array(self.initial_params)
            + 1e-4 * np.random.randn(self.num_walkers, self.num_params)
        )  # TODO(AstroKriel): Move from np.random.randn to np.random.generator. Define self.rng: Generator in __init__ then use self.rng.randn
        mcmc_sampler = emcee.EnsembleSampler(
            nwalkers=self.num_walkers,
            ndim=self.num_params,
            log_prob_fn=self._log_posterior,
            vectorize=True,
        )

        # TODO(AstroKriel): Add comment here describing how this works
        deque(
            tqdm(
                mcmc_sampler.sample(
                    initial_state=perturbed_params, iterations=num_steps
                ),
                total=num_steps,
            ),
            maxlen=0,  # discard returned samples; deque is only used to force evaluation of the generator
        )

        # save key outputs
        self.raw_chain = mcmc_sampler.get_chain()
        self._check_chain_convergence(mcmc_sampler)
        self.fitted_posterior_samples = mcmc_sampler.get_chain(
            discard=burn_in_steps, thin=10, flat=True
        )
        self.output_posterior_samples, self.output_param_labels = (
            self._get_output_params()
        )

        # TODO(AstroKriel): Add comment describing how this works
        # TODO(AstroKriel): print -> logging
        if np.array_equal(self.output_posterior_samples, self.fitted_posterior_samples):
            print("Estimating the KDE of only the fitted posterior...")
            self.fitted_posterior_kde = gaussian_kde(
                self.fitted_posterior_samples.T, bw_method="scott"
            )
            self.output_posterior_kde = self.fitted_posterior_kde
        else:
            print("Estimating the KDE of both the fitted and output posteriors...")
            self.fitted_posterior_kde = gaussian_kde(
                self.fitted_posterior_samples.T, bw_method="scott"
            )
            self.output_posterior_kde = gaussian_kde(
                self.output_posterior_samples.T, bw_method="scott"
            )
        # create diagnostic outputs
        # TODO(AstroKriel): Might be worth making these optional. That way if you want to test a change to your fitting you don't need to wait for plotting as well
        self._make_plots()
        self._save_samples_to_disk()

    def _log_posterior(
        self, param_vectors: "npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.float64]":
        lp_values = self._log_prior(param_vectors)
        valid_prior_mask = np.isfinite(lp_values)
        ll_values = np.full_like(lp_values, -np.inf)
        ll_values[valid_prior_mask] = self._log_likelihood(
            param_vectors[valid_prior_mask]
        )
        return lp_values + ll_values

    def _log_prior(
        self, param_vectors: "npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.float64]":
        valid_params_mask = self._get_valid_params_mask(param_vectors)
        num_local_walkers = param_vectors.shape[0]
        lp_values = np.full(num_local_walkers, -np.inf)
        if self._prior_logpdf is not None:
            valid_params = np.atleast_2d(param_vectors[valid_params_mask])
            kde_vector = self._get_kde_params(valid_params)
            kde_logpdfs = self._prior_logpdf(kde_vector.T)
            lp_values[valid_params_mask] = kde_logpdfs
        else:
            lp_values[valid_params_mask] = 0.0  # uniform prior
        return lp_values

    def _log_likelihood(
        self, param_vectors: "npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.float64]":
        param_vectors = np.atleast_2d(param_vectors)
        num_local_walkers = param_vectors.shape[0]
        valid_params_mask = self._get_valid_params_mask(param_vectors)
        ll_values = np.full(num_local_walkers, -np.inf)
        if not np.any(valid_params_mask):
            return ll_values
        try:
            valid_params = param_vectors[valid_params_mask]
            modelled_y = self._model(valid_params)
            measured_y = np.asarray(self.y_values)
            # add a leading dimension to the measured data so it broadcasts across the vectorised param-rows
            y_residuals = measured_y[None, :] - modelled_y
            ll_values[valid_params_mask] = -0.5 * np.sum(
                np.square(y_residuals / self.likelihood_sigma), axis=1
            )
        # TODO(AstroKriel): Don't blindly catch exceptions!!!! Figure out why it crashes, and capture and resolve it properly
        # This is functionally equivalent to not having a try / except clause at all, since any crash is immediately raised anyway
        except Exception as error:
            raise
        return ll_values

    # TODO(AstroKriel): print -> logging
    def _check_chain_convergence(self, mcmc_sampler: emcee.EnsembleSampler) -> None:
        try:
            self.auto_correlation_time = mcmc_sampler.get_autocorr_time()
            converged = np.all(self.auto_correlation_time * 50 < self.num_steps)
            if np.any(self.auto_correlation_time * 5 > self.num_steps):
                print(
                    "WARNING: Chain length may be too short to reliably estimate the autocorrelation time."
                )
            if not converged:
                print("WARNING: Chain appears to not have converged.")
            else:
                print(
                    f"Chains appear to have converged. The autocorrelation time for the parameters are: {self.auto_correlation_time}"
                )
        except emcee.autocorr.AutocorrError:
            print(
                "WARNING: The autocorrelation time could not be estimated reliably. Chain may not be long enough."
            )

    def _make_plots(self) -> None:
        plot_chain_evolution.PlotChainEvolution(self).plot()
        plot_model_posteriors.PlotModelPosteriors(self).plot()
        plot_model_fits.PlotModelFits(self).plot()

    def _save_samples_to_disk(self) -> None:
        fitted_posterior_path = io_manager.combine_file_path_parts([
            self.output_directory,
            f"{self.routine_name}_fitted_posterior_samples.npy",
        ])
        output_posterior_path = io_manager.combine_file_path_parts([
            self.output_directory,
            f"{self.routine_name}_output_posterior_samples.npy",
        ])
        np.save(fitted_posterior_path, self.fitted_posterior_samples)
        if not np.array_equal(
            self.output_posterior_samples, self.fitted_posterior_samples
        ):
            np.save(output_posterior_path, self.output_posterior_samples)


# END OF MODULE
