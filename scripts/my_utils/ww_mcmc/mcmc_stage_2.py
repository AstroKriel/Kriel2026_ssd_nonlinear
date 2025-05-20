## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from . import base_mcmc


## ###############################################################
## HELPER FUNCTION
## ###############################################################

def plot_param_percentiles(ax, samples, orientation):
  p16, p50, p84 = numpy.percentile(samples, [16, 50, 84])
  if   "h" in orientation.lower():
    ax_line = ax.axhline
    ax_span = ax.axhspan
  elif "v" in orientation.lower():
    ax_line = ax.axvline
    ax_span = ax.axvspan
  else: raise ValueError("`orientation` must either be `horizontal` (`h`) or `vertical` (`v`).")
  ax_line(p50, color="green", ls=":", lw=1.5, zorder=5)
  ax_span(p16, p84, color="green", ls="-", lw=1.5, alpha=0.3, zorder=4)


## ###############################################################
## STAGE 2 MCMC FITTER
## ###############################################################

class MCMCStage2Routine(base_mcmc.BaseMCMCRoutine):
  def __init__(
      self,
      *,
      output_directory : str,
      x_values         : list | numpy.ndarray,
      y_values         : list | numpy.ndarray,
      initial_params   : tuple[float, ...],
      likelihood_sigma : float = 1.0,
      prior_kde        = None,
      verbose          : bool = True,
      debug_mode       : bool = False
    ):
    self.max_time = numpy.max(x_values)
    super().__init__(
      output_directory    = output_directory,
      routine_name        = "stage2",
      verbose             = verbose,
      debug_mode          = debug_mode,
      x_values            = x_values,
      y_values            = y_values,
      prior_kde           = prior_kde,
      likelihood_sigma    = likelihood_sigma,
      initial_params      = initial_params,
      y_data_label        = r"$E_{\mathrm{mag}}$",
      fitted_param_labels = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$\log_{10}(E_{\mathrm{sat}})$",
        r"$\gamma$",
        r"$t_{\mathrm{nl}}$",
        r"$t_{\mathrm{sat}}$",
      ]
    )

  def _model(self, fit_params):
    fit_params = numpy.atleast_2d(fit_params)
    log10_init_energy, log10_sat_energy, gamma, start_nl_time, start_sat_time = fit_params.T
    x_vals             = self.x_values
    n_walkers          = fit_params.shape[0]
    n_times            = x_vals.shape[0]
    x_vals_2d          = x_vals[None, :]
    start_nl_time_2d   = start_nl_time[:, None]
    start_sat_time_2d  = start_sat_time[:, None]
    gamma_2d           = gamma[:, None]
    mask_exp_phase     = x_vals_2d < start_nl_time_2d
    mask_nl_phase      = (start_nl_time_2d <= x_vals_2d) & (x_vals_2d < start_sat_time_2d)
    mask_sat_phase     = start_sat_time_2d < x_vals_2d
    init_energy        = 10**log10_init_energy
    sat_energy         = 10**log10_sat_energy
    start_nl_energy    = init_energy * numpy.exp(gamma * start_nl_time)
    alpha              = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
    start_nl_energy_2d = start_nl_energy[:, None]
    alpha_2d           = alpha[:, None]
    energy             = numpy.zeros((n_walkers, n_times))
    energy[mask_exp_phase] = (init_energy[:, None] * numpy.exp(gamma_2d * x_vals_2d))[mask_exp_phase]
    energy[mask_nl_phase]  = (start_nl_energy_2d + alpha_2d * (x_vals_2d - start_nl_time_2d))[mask_nl_phase]
    energy[mask_sat_phase] = numpy.broadcast_to(sat_energy[:, None], (n_walkers, n_times))[mask_sat_phase]
    return energy


  def _check_params_are_valid(self, fit_params):
    fit_params = numpy.atleast_2d(fit_params)
    log10_init_energy   = fit_params[:, 0]
    log10_sat_energy    = fit_params[:, 1]
    gamma               = fit_params[:, 2]
    start_nl_time       = fit_params[:, 3]
    start_sat_time      = fit_params[:, 4]
    cond_init_energy    = (-30 < log10_init_energy) & (log10_init_energy < -5)
    cond_sat_energy     = (-5 < log10_sat_energy) & (log10_sat_energy < 0)
    cond_gamma          = (0 < gamma) & (gamma < 2)
    cond_start_nl_time  = (0.1 * self.max_time < start_nl_time) & (start_nl_time < start_sat_time)
    cond_start_sat_time = start_sat_time < self.max_time
    valid = cond_init_energy & cond_sat_energy & cond_gamma & cond_start_nl_time & cond_start_sat_time
    if fit_params.shape[0] == 1:
      return valid[0]
    return valid

  def _get_kde_params(self, param_vectors):
    ## ignore transition times: use a unifrom prior for them
    return numpy.asarray(param_vectors[:, :3])

  def _annotate_fitted_params(self, axs):
    gamma_samples             = self.fitted_posterior_samples[:,2]
    start_nl_time_samples     = self.fitted_posterior_samples[:,3]
    start_sat_time_samples    = self.fitted_posterior_samples[:,4]
    init_energy_samples       = 10**self.fitted_posterior_samples[:,0]
    sat_energy_samples        = 10**self.fitted_posterior_samples[:,1]
    start_nl_energy_samples   = init_energy_samples * numpy.exp(gamma_samples * start_nl_time_samples)
    alpha_samples             = (sat_energy_samples - start_nl_energy_samples) / (start_sat_time_samples - start_nl_time_samples)
    plot_param_percentiles(axs[0], sat_energy_samples, orientation="horizontal")
    plot_param_percentiles(axs[1], alpha_samples, orientation="horizontal")
    for row_index in range(len(axs)):
      plot_param_percentiles(axs[row_index], start_nl_time_samples, orientation="vertical")
      plot_param_percentiles(axs[row_index], start_sat_time_samples, orientation="vertical")


## END OF MODULE