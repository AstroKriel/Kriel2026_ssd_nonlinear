## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from . import base_mcmc
from . import mcmc_utils


## ###############################################################
## STAGE 1 MCMC FITTER
## ###############################################################

class Stage1MCMCRoutine(base_mcmc.BaseMCMCRoutine):
  def __init__(
      self,
      *,
      output_directory        : str,
      time_values             : list | numpy.ndarray,
      ave_log10_energy_values : list | numpy.ndarray,
      std_log10_energy_values : list | numpy.ndarray,
      initial_params          : tuple[float, ...],
      plot_posterior_kde      : bool = True
    ):
    self.log10_e  = numpy.log10(numpy.exp(1))
    self.max_time = numpy.max(time_values)
    super().__init__(
      routine_name        = "stage1",
      output_directory    = output_directory,
      x_values            = time_values,
      y_values            = ave_log10_energy_values,
      likelihood_sigma    = std_log10_energy_values,
      initial_params      = initial_params,
      plot_posterior_kde  = plot_posterior_kde,
      data_label          = r"$\log_{10}(E_{\mathrm{mag}})$",
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
    valid_gamma             = (0 < gamma) & (gamma < 10)
    valid_transition_time   = (0.1 * self.max_time < transition_time) & (transition_time < 0.9 * self.max_time)
    valid_params_mask       = valid_log10_init_energy & valid_gamma & valid_transition_time
    if num_local_walkers == 1:
      return valid_params_mask[0]
    return valid_params_mask

  def _annotate_fitted_params(self, axs):
    log10_gamma_samples     = self.log10_e * self.fitted_posterior_samples[:,1]
    transition_time_samples = self.fitted_posterior_samples[:,2]
    mcmc_utils.plot_param_percentiles(axs[1], log10_gamma_samples, orientation="horizontal")
    for row_index in range(len(axs)):
      mcmc_utils.plot_param_percentiles(axs[row_index], transition_time_samples, orientation="vertical")

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
    mcmc_utils.plot_param_percentiles(axs[0], log10_sat_energy_samples, orientation="horizontal")


## END OF MODULE