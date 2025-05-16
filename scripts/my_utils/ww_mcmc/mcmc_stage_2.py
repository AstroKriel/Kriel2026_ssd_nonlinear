## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from jormi.utils import list_utils
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager
from . import base_mcmc


## ###############################################################
## STAGE 2 MCMC FITTER
## ###############################################################

class MCMCStage2Routine(base_mcmc.BaseMCMCRoutine):
  def __init__(self, output_directory, x_values, y_values, verbose):
    self.max_time = numpy.max(x_values)
    super().__init__(
      output_directory = output_directory,
      routine_name     = "stage2",
      verbose          = verbose,
      x_values         = x_values,
      y_values         = y_values,
      likelihood_sigma = 0.5 * y_values[-1],
      initial_params   = (0.85 * self.transition_time, 1.25 * self.transition_time, log10_sat_energy_guess),
      y_data_label     = r"$\log_{10}(E_{\mathrm{mag}})$",
      fitted_param_labels     = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$\gamma$",
        r"$t_{\mathrm{nl}}$",
        r"$t_{\mathrm{sat}}$",
        r"$\log_{10}(E_{\mathrm{sat}})$"
      ]
    )

  def _model(self, fit_params):
    (start_nl_time, start_sat_time, log10_sat_energy) = fit_params
    ## mask different ssd phases
    mask_exp_phase = self.x_values < start_nl_time
    mask_nl_phase  = (start_nl_time <= self.x_values) & (self.x_values < start_sat_time)
    mask_sat_phase = start_sat_time < self.x_values
    ## compute model constants
    start_nl_energy = self.init_energy * numpy.exp(self.gamma * start_nl_time)
    sat_energy      = 10**log10_sat_energy
    alpha           = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
    ## model energy evolution
    energy = numpy.zeros_like(self.x_values)
    energy[mask_exp_phase] = self.init_energy * numpy.exp(self.gamma * self.x_values[mask_exp_phase])
    energy[mask_nl_phase]  = start_nl_energy + alpha * (self.x_values[mask_nl_phase] - start_nl_time)
    energy[mask_sat_phase] = sat_energy
    return energy

  def _check_params_are_valid(self, fit_params, print_errors=False):
    (start_nl_time, start_sat_time, log10_sat_energy) = fit_params
    errors = []
    if not (0.1 * self.max_time < start_nl_time < self.transition_time):
      errors.append(f"`start_nl_time` ({start_nl_time:.2f}) must be larger than 0.1 * `max_time` ({self.max_time:.2f}) and smaller than the stage-1 estimated transition x_values ({self.transition_time:.2f}).")
    if not (self.transition_time < start_sat_time < self.max_time):
      errors.append(f"`start_sat_time` ({start_sat_time:.2f}) must be larger than the stage-1 estimated transition x_values ({self.transition_time:.2f}) and less than `max_time` ({self.max_time:.2f}).")
    if not (-5 < log10_sat_energy < 0):
      errors.append(f"`log10_sat_energy` ({log10_sat_energy:.2f}) must be between -5 and 0.")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True


## END OF MODULE