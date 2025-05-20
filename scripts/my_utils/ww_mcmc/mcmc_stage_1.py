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
## STAGE 1 MCMC FITTER
## ###############################################################

class MCMCStage1Routine(base_mcmc.BaseMCMCRoutine):
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
    self.log10_e  = numpy.log10(numpy.exp(1))
    self.max_time = numpy.max(x_values)
    super().__init__(
      output_directory    = output_directory,
      routine_name        = "stage1",
      verbose             = verbose,
      plot_kde            = plot_kde,
      x_values            = x_values,
      y_values            = numpy.log10(y_values),
      likelihood_sigma    = likelihood_sigma,
      initial_params      = initial_params,
      prior_kde           = prior_kde,
      y_label             = r"$\log_{10}(E_{\mathrm{mag}})$",
      fitted_param_labels = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$\gamma$",
        r"$t_{\mathrm{approx}}$",
      ]
    )

  def _model(self, param_vectors):
    param_vectors = numpy.atleast_2d(param_vectors) # (N, P)
    ## output dimensions
    num_local_walkers = param_vectors.shape[0] # N
    num_data_points = len(self.x_values) # T
    ## unpack model parameters (P = 3)
    log10_init_energy, gamma, transition_time = param_vectors.T
    ## reshape parameters to allow for vectorising over param-rows
    x_values_2d          = self.x_values[None, :] # shape (1, T)
    gamma_2d             = gamma[:, None] # shape (N, 1)
    transition_time_2d   = transition_time[:, None] # shape (N, 1)
    log10_init_energy_2d = log10_init_energy[:, None] # shape (N, 1)
    ## mask (reduced) SSD phases
    mask_exp_phase = x_values_2d < transition_time_2d
    mask_sat_phase = ~mask_exp_phase
    ## compute (reduced) SSD phases
    log10_energy_exp_values = log10_init_energy_2d + self.log10_e * gamma_2d * x_values_2d # (N, T)
    log10_energy_sat_values = log10_init_energy_2d + self.log10_e * gamma_2d * transition_time_2d # (N, 1)
    log10_energy_sat_values = numpy.broadcast_to(log10_energy_sat_values, (num_local_walkers, num_data_points)) # (N, T)
    ## assemble modelled (reduced) SSD phases
    log10_energy = numpy.empty((num_local_walkers, num_data_points))
    log10_energy[mask_exp_phase] = log10_energy_exp_values[mask_exp_phase]
    log10_energy[mask_sat_phase] = log10_energy_sat_values[mask_sat_phase]
    return log10_energy

  def _get_valid_params_mask(self, param_vectors):
    param_vectors = numpy.atleast_2d(param_vectors)
    num_local_walkers = param_vectors.shape[0]
    log10_init_energy, gamma, transition_time = param_vectors.T
    valid_log10_init_energy = (-30 < log10_init_energy) & (log10_init_energy < -5)
    valid_gamma             = (0 < gamma) & (gamma < 2)
    valid_transition_time   = (0.25 * self.max_time < transition_time) & (transition_time < 0.9 * self.max_time)
    valid_params_mask       = valid_log10_init_energy & valid_gamma & valid_transition_time
    if num_local_walkers == 1:
      return valid_params_mask[0]
    return valid_params_mask

  def _annotate_fitted_params(self, axs):
    log10_gamma_samples     = self.log10_e * self.fitted_posterior_samples[:,1]
    transition_time_samples = self.fitted_posterior_samples[:,2]
    plot_param_percentiles(axs[1], log10_gamma_samples, orientation="horizontal")
    for row_index in range(len(axs)):
      plot_param_percentiles(axs[row_index], transition_time_samples, orientation="vertical")

  def _get_output_params(self):
    log10_init_energy_samples = self.fitted_posterior_samples[:,0]
    gamma_samples             = self.fitted_posterior_samples[:,1]
    transition_time_samples   = self.fitted_posterior_samples[:,2]
    log10_sat_energy_samples  = log10_init_energy_samples + self.log10_e * gamma_samples * transition_time_samples
    output_param_samples = numpy.column_stack([
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

  def _annotate_output_params(self, axs):
    log10_sat_energy_samples = self.output_posterior_samples[:,1]
    plot_param_percentiles(axs[0], log10_sat_energy_samples, orientation="horizontal")


## END OF MODULE