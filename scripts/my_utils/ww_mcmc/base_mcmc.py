## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
import emcee
from pathlib import Path
from scipy.stats import gaussian_kde
from . import plot_chain_evolution
from . import plot_model_posteriors
from . import plot_model_fits


## ###############################################################
## BASE ROUTINE
## ###############################################################

class BaseMCMCRoutine:

  def _model(self, param_vector: tuple[float, ...]):
    raise NotImplementedError()

  def _check_params_are_valid(self, param_vector: tuple[float, ...], print_errors: bool = False):
    raise NotImplementedError()

  def _annotate_fit(self, axs):
    pass

  def __init__(
      self,
      output_directory : str | Path,
      routine_name     : str,
      x_values         : list | numpy.ndarray,
      y_values         : list | numpy.ndarray,
      initial_params   : tuple[float, ...],
      likelihood_sigma : float | list | numpy.ndarray = 1.0,
      y_data_label     : str | None = None,
      param_labels     : list[str] = [],
      verbose          : bool = True
    ):
    self.output_directory = output_directory
    self.routine_name     = routine_name
    self.x_values         = numpy.asarray(x_values)
    self.y_values         = numpy.asarray(y_values)
    self.initial_params   = initial_params
    self.num_params       = len(self.initial_params)
    self.likelihood_sigma = likelihood_sigma
    self.y_data_label     = y_data_label
    self.param_labels     = param_labels
    self.verbose          = verbose
    self._validate_inputs()
    self.posterior_samples = None
    self.posterior_kde     = None

  def _validate_inputs(self):
    if not isinstance(self.x_values, (list, numpy.ndarray)):
      raise ValueError(f"`x_values` should be either a list or array of values.")
    if not isinstance(self.y_values, (list, numpy.ndarray)):
      raise ValueError(f"`y_values` should be either a list or array of values.")
    if len(self.x_values) != len(self.y_values):
      raise ValueError(f"`x_values` and `y_values` should be the same length, but got {len(self.x_values)} vs {len(self.y_values)}.")
    if not isinstance(self.likelihood_sigma, (float, int)):
      raise ValueError(f"`likelihood_sigma` should be a scalar.")
    self.likelihood_sigma = float(self.likelihood_sigma)

  def sample_posterior(
      self,
      num_walkers   : int = 200,
      num_steps     : int = 5000,
      burn_in_steps : int = 2000,
    ):
    if not self._check_params_are_valid(self.initial_params, print_errors=True):
      raise ValueError("Initial guess is invalid!")
    print("Estimating parameters...")
    self.num_walkers = num_walkers
    perturbed_params = numpy.array(self.initial_params) + 1e-4 * numpy.random.randn(self.num_walkers, self.num_params)
    mcmc_sampler = emcee.EnsembleSampler(self.num_walkers, self.num_params, self._log_posterior)
    mcmc_sampler.run_mcmc(perturbed_params, num_steps)
    self.raw_chain = mcmc_sampler.get_chain()
    self.posterior_samples = mcmc_sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    self._compute_posterior_kde()
    plot_chain_evolution.PlotChainEvolution(self).plot()
    # plot_model_posteriors.PlotModelPosteriors(self).plot()
    plot_model_fits.PlotModelFits(self).plot()

  def print_log_likelihood(self, param_vector):
    ll_value = self._log_likelihood(param_vector)
    print(f"param_vector = ({param_vector}) yields log-likelihood = {ll_value:.2e}")

  def _log_prior(self, param_vector):
    if not self._check_params_are_valid(param_vector):
      return -numpy.inf
    return 0

  def _log_likelihood(self, param_vector):
    if not self._check_params_are_valid(param_vector):
      return -numpy.inf
    try:
      residual = self.y_values - self._model(param_vector)
      ll_value = -0.5 * numpy.sum(numpy.square(residual / self.likelihood_sigma))
      if not numpy.isfinite(ll_value):
        return -numpy.inf
      return ll_value
    except Exception as error:
      print("Error in likelihood:", error, param_vector)
      return -numpy.inf

  def _log_posterior(self, param_vector):
    lp_value = self._log_prior(param_vector)
    if not numpy.isfinite(lp_value): return -numpy.inf
    ll_value = self._log_likelihood(param_vector)
    return lp_value + ll_value

  def _compute_posterior_kde(self):
    print("Estimating KDE of posterior samples...")
    self.posterior_kde = gaussian_kde(self.posterior_samples.T, bw_method="scott")


## END OF MODULE