## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
import emcee
from tqdm import tqdm
from pathlib import Path
from collections import deque
from scipy.stats import gaussian_kde
from . import plot_chain_evolution
from . import plot_model_posteriors
from . import plot_model_fits


## ###############################################################
## BASE ROUTINE
## ###############################################################

class BaseMCMCRoutine:
  """
  Base class for running MCMC inference using emcee. 
  Subclasses must define a model and parameter validation logic.
  """

  ## abstract methods that need to be implemented by subclasses

  def _model(self, param_vectors):
    raise NotImplementedError()

  def _get_valid_params_mask(self, param_vectors):
    raise NotImplementedError()

  def _get_kde_params(self, param_vectors):
    return numpy.asarray(param_vectors)

  def _get_output_params(self):
    return self.fitted_posterior_samples, self.fitted_param_labels

  def _annotate_fitted_params(self, axs):
    pass

  def _annotate_output_params(self, axs):
    pass

  ## core functionality

  def __init__(
      self,
      *,
      output_directory    : str | Path,
      routine_name        : str,
      x_values            : list | numpy.ndarray,
      y_values            : list | numpy.ndarray,
      initial_params      : tuple[float, ...],
      prior_kde           : callable = None,
      likelihood_sigma    : float = 1.0,
      y_label             : str | None = None,
      fitted_param_labels : list[str] = [],
      verbose             : bool = True,
      plot_kde            : bool = False,
    ):
    self.output_directory    = output_directory
    self.routine_name        = routine_name
    self.x_values            = numpy.asarray(x_values)
    self.y_values            = numpy.asarray(y_values)
    self.initial_params      = initial_params
    self.num_params          = len(self.initial_params)
    self.prior_kde           = prior_kde
    self.likelihood_sigma    = likelihood_sigma
    self.y_label             = y_label
    self.fitted_param_labels = fitted_param_labels
    self.verbose             = verbose
    self.plot_kde            = plot_kde
    self._validate_inputs()
    ## define key outputs
    self.raw_chain                = None
    self.auto_correlation_time    = None
    self.fitted_posterior_samples = None
    self.fitted_posterior_kde     = None
    self.output_posterior_samples = None
    self.output_posterior_kde     = None

  def _validate_inputs(self):
    if not isinstance(self.x_values, (list, numpy.ndarray)): raise ValueError(f"`x_values` should either be a list or array of values.")
    if not isinstance(self.y_values, (list, numpy.ndarray)): raise ValueError(f"`y_values` should either be a list or array of values.")
    if len(self.x_values) != len(self.y_values): raise ValueError(f"`x_values` and `y_values` should be the same length. Received {len(self.x_values)} vs {len(self.y_values)}.")
    if not isinstance(self.initial_params, tuple): raise ValueError(f"`initial_params` should be a tuple. Received {type(self.initial_params)}.")
    if not isinstance(self.likelihood_sigma, float): raise ValueError(f"`likelihood_sigma` should be a scalar.")

  def estimate_posterior(
      self,
      num_walkers   : int = 200,
      num_steps     : int = 5000,
      burn_in_steps : int = 1000,
    ):
    if not self._get_valid_params_mask(self.initial_params):
      raise ValueError("Initial guess is invalid!")
    print("Estimating the posterior...")
    self.num_walkers = num_walkers
    self.num_steps   = num_steps
    perturbed_params = numpy.array(self.initial_params) + 1e-4 * numpy.random.randn(self.num_walkers, self.num_params)
    mcmc_sampler = emcee.EnsembleSampler(
      nwalkers    = num_walkers,
      ndim        = self.num_params,
      log_prob_fn = self._log_posterior,
      vectorize   = True
    )
    deque(
      tqdm(
        mcmc_sampler.sample(initial_state=perturbed_params, iterations=num_steps),
        total = num_steps
      ),
      maxlen = 0 # discard returned samples; deque is only used to force evaluation of the generator
    )
    ## save key outputs
    self.raw_chain = mcmc_sampler.get_chain()
    self._check_chain_convergence(mcmc_sampler)
    self.fitted_posterior_samples = mcmc_sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    self.output_posterior_samples, self.output_param_labels = self._get_output_params()
    if numpy.array_equal(self.output_posterior_samples, self.fitted_posterior_samples):
      print("Estimating the KDE of only the fitted posterior...")
      self.fitted_posterior_kde = gaussian_kde(self.fitted_posterior_samples.T, bw_method="scott")
      self.output_posterior_kde = self.fitted_posterior_kde
    else:
      print("Estimating the KDE of both the fitted and output posteriors...")
      self.fitted_posterior_kde = gaussian_kde(self.fitted_posterior_samples.T, bw_method="scott")
      self.output_posterior_kde = gaussian_kde(self.output_posterior_samples.T, bw_method="scott")
    ## create diagnostic plots
    self._make_plots()

  def _log_posterior(self, param_vectors):
    lp_values = self._log_prior(param_vectors)
    valid_prior_mask = numpy.isfinite(lp_values)
    ll_values = numpy.full_like(lp_values, -numpy.inf)
    ll_values[valid_prior_mask] = self._log_likelihood(param_vectors[valid_prior_mask])
    return lp_values + ll_values

  def _log_prior(self, param_vectors):
    valid_params_mask = self._get_valid_params_mask(param_vectors)
    num_local_walkers = param_vectors.shape[0]
    lp_values = numpy.full(num_local_walkers, -numpy.inf)
    if self.prior_kde is not None:
      valid_params = numpy.atleast_2d(param_vectors[valid_params_mask])
      kde_vector   = self._get_kde_params(valid_params)
      kde_logpdfs  = self.prior_kde.logpdf(kde_vector.T)
      lp_values[valid_params_mask] = kde_logpdfs
    else: lp_values[valid_params_mask] = 0.0 # uniform prior
    return lp_values

  def _log_likelihood(self, param_vectors):
    param_vectors = numpy.atleast_2d(param_vectors)
    num_local_walkers = param_vectors.shape[0]
    valid_params_mask = self._get_valid_params_mask(param_vectors)
    ll_values = numpy.full(num_local_walkers, -numpy.inf)
    if not numpy.any(valid_params_mask):
      return ll_values
    try:
      valid_params = param_vectors[valid_params_mask]
      modelled_y   = self._model(valid_params)
      measured_y   = numpy.asarray(self.y_values)
      ## add leading dimension to measured data so it broadcasts across vectorised param-rows
      y_residuals  = measured_y[None, :] - modelled_y
      ll_values[valid_params_mask] = -0.5 * numpy.sum(numpy.square(y_residuals / self.likelihood_sigma), axis=1)
    except Exception as error:
      raise
    return ll_values

  def _check_chain_convergence(self, mcmc_sampler):
    try:
      self.auto_correlation_time = mcmc_sampler.get_autocorr_time()
      converged = numpy.all(self.auto_correlation_time * 50 < self.num_steps)
      if numpy.any(self.auto_correlation_time * 5 > self.num_steps):
        print("WARNING: Chain length may be too short to reliably estimate the autocorrelation time.")
      if not converged:
        print("WARNING: Chain appears to not have converged.")
      else: print(f"Chains appear to have converged. The autocorrelation time for the parameters are: {self.auto_correlation_time}")
    except emcee.autocorr.AutocorrError:
      print("WARNING: The autocorrelation time could not be estimated reliably. Chain may not be long enough.")

  def _make_plots(self):
    plot_chain_evolution.PlotChainEvolution(self).plot()
    plot_model_posteriors.PlotModelPosteriors(self).plot()
    plot_model_fits.PlotModelFits(self).plot()


## END OF MODULE