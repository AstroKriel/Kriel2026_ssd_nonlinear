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
      prior_kde        : callable = None,
      verbose          : bool = True,
      plot_kde         : bool = False
    ):
    self.max_time = numpy.max(x_values)
    super().__init__(
      output_directory    = output_directory,
      routine_name        = "stage2",
      verbose             = verbose,
      plot_kde            = plot_kde,
      x_values            = x_values,
      y_values            = y_values,
      prior_kde           = prior_kde,
      likelihood_sigma    = likelihood_sigma,
      initial_params      = initial_params,
      y_label             = r"$E_{\mathrm{mag}}$",
      fitted_param_labels = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$\log_{10}(E_{\mathrm{sat}})$",
        r"$\gamma$",
        r"$t_{\mathrm{nl}}$",
        r"$t_{\mathrm{sat}}$",
      ]
    )

  def _model(self, param_vectors):
    param_vectors = numpy.atleast_2d(param_vectors) # (N, P)
    ## output dimensions
    num_local_walkers = param_vectors.shape[0] # N
    num_data_points = len(self.x_values) # T
    ## unpack model parameters (P = 5)
    log10_init_energy, log10_sat_energy, gamma, start_nl_time, start_sat_time = param_vectors.T
    ## reshape parameters to allow for vectorising over param-rows
    x_values_2d        = self.x_values[None, :] # shape (1, T)
    start_nl_time_2d   = start_nl_time[:, None] # shape (N, 1)
    start_sat_time_2d  = start_sat_time[:, None] # shape (N, 1)
    gamma_2d           = gamma[:, None] # shape (N, 1)
    ## mask SSD phases
    mask_exp_phase     = x_values_2d < start_nl_time_2d
    mask_nl_phase      = (start_nl_time_2d <= x_values_2d) & (x_values_2d < start_sat_time_2d)
    mask_sat_phase     = start_sat_time_2d < x_values_2d
    ## compute model constants (per walker)
    init_energy        = 10**log10_init_energy # (N,)
    sat_energy         = 10**log10_sat_energy # (N,)
    start_nl_energy    = init_energy * numpy.exp(gamma * start_nl_time) # (N,)
    alpha              = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time) # (N,)
    init_energy_2d     = init_energy[:, None] # (N, 1)
    sat_energy_2d      = sat_energy[:, None] # (N, 1)
    start_nl_energy_2d = start_nl_energy[:, None] # (N, 1)
    alpha_2d           = alpha[:, None] # (N, 1)
    ## assemble modelled SSD phases
    energy = numpy.zeros((num_local_walkers, num_data_points))
    energy[mask_exp_phase] = (init_energy_2d * numpy.exp(gamma_2d * x_values_2d))[mask_exp_phase] # (N, T)
    energy[mask_nl_phase]  = (start_nl_energy_2d + alpha_2d * (x_values_2d - start_nl_time_2d))[mask_nl_phase] # (N, T)
    energy[mask_sat_phase] = numpy.broadcast_to(sat_energy_2d, (num_local_walkers, num_data_points))[mask_sat_phase] # (N, T)
    return energy

  def _get_valid_params_mask(self, param_vectors):
    param_vectors = numpy.atleast_2d(param_vectors)
    num_local_walkers = param_vectors.shape[0]
    log10_init_energy, log10_sat_energy, gamma, start_nl_time, start_sat_time = param_vectors.T
    valid_log10_init_energy = (-30 < log10_init_energy) & (log10_init_energy < -5)
    valid_log10_sat_energy  = (-5 < log10_sat_energy) & (log10_sat_energy < 0)
    valid_gamma             = (0 < gamma) & (gamma < 2)
    valid_start_nl_time     = (0.1 * self.max_time < start_nl_time) & (start_nl_time < start_sat_time)
    valid_start_sat_time    = start_sat_time < self.max_time
    valid_params_mask = (
      valid_log10_init_energy & valid_log10_sat_energy & valid_gamma & valid_start_nl_time & valid_start_sat_time
    )
    if num_local_walkers == 1:
      return valid_params_mask[0]
    return valid_params_mask

  def _get_kde_params(self, param_vectors):
    ## ignore the transition times; use a unifrom prior for them
    return numpy.asarray(param_vectors[:, :3])

  def _annotate_fitted_params(self, axs):
    init_energy_samples     = 10**self.fitted_posterior_samples[:,0]
    sat_energy_samples      = 10**self.fitted_posterior_samples[:,1]
    gamma_samples           = self.fitted_posterior_samples[:,2]
    start_nl_time_samples   = self.fitted_posterior_samples[:,3]
    start_sat_time_samples  = self.fitted_posterior_samples[:,4]
    start_nl_energy_samples = init_energy_samples * numpy.exp(gamma_samples * start_nl_time_samples)
    alpha_samples           = (sat_energy_samples - start_nl_energy_samples) / (start_sat_time_samples - start_nl_time_samples)
    plot_param_percentiles(axs[0], sat_energy_samples, orientation="horizontal")
    plot_param_percentiles(axs[1], alpha_samples, orientation="horizontal")
    for row_index in range(len(axs)):
      plot_param_percentiles(axs[row_index], start_nl_time_samples, orientation="vertical")
      plot_param_percentiles(axs[row_index], start_sat_time_samples, orientation="vertical")


## END OF MODULE