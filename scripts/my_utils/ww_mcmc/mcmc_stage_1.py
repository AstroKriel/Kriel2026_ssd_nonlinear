## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from . import base_mcmc


## ###############################################################
## STAGE 1 MCMC FITTER
## ###############################################################

class MCMCStage1Routine(base_mcmc.BaseMCMCRoutine):
  def __init__(self, output_directory, x_values, y_values, verbose):
    self.log10_e = numpy.log10(numpy.exp(1))
    self.max_time = numpy.max(x_values)
    super().__init__(
      output_directory = output_directory,
      routine_name     = "stage1",
      verbose          = verbose,
      x_values         = x_values,
      y_values         = numpy.log10(y_values),
      initial_params   = (-20, 0.85 * numpy.max(x_values), 0.5),
      y_data_label     = r"$\log_{10}(E_{\mathrm{mag}})$",
      param_labels     = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$t_{\mathrm{approx}}$",
        r"$\gamma$"
      ]
    )

  def _model(self, param_vector):
    (log10_init_energy, transition_time, gamma) = param_vector
    mask_exp = self.x_values < transition_time
    mask_sat = ~mask_exp
    log10_energy = numpy.zeros_like(self.x_values)
    log10_energy[mask_exp] = log10_init_energy + self.log10_e * gamma * self.x_values[mask_exp]
    log10_energy[mask_sat] = log10_init_energy + self.log10_e * gamma * transition_time
    return log10_energy

  def _check_params_are_valid(self, param_vector, print_errors=False):
    (log10_init_energy, transition_time, gamma) = param_vector
    errors = []
    if not (-30 < log10_init_energy < -5):
      errors.append(f"`log10_init_energy` ({log10_init_energy:.2f}) must be between -20 and -5.")
    if not (0.25 * self.max_time < transition_time < 0.9 * self.max_time):
      errors.append(f"`transition_time` ({transition_time:.2f}) must be between 25 and 90 percent of `max_time` ({self.max_time:.2f}).")
    if not (0 < gamma < 2):
      errors.append(f"`gamma` ({gamma:.2f}) must be between 0 and 2.")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True

  def _annotate_fit(self, axs):
    gamma_samples = self.posterior_samples[:,2]
    gamma_p16, gamma_p50, gamma_p84 = numpy.percentile(gamma_samples, [16, 50, 84])
    axs[1].axhspan(self.log10_e * gamma_p16, self.log10_e * gamma_p84, color="red", ls="-", lw=1.5, alpha=0.3)
    axs[1].axhline(self.log10_e * gamma_p50, color="red", ls=":", lw=1.5)


## END OF MODULE