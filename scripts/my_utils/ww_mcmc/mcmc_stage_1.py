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
  ax_line(p50, color="black", ls=":", lw=1.5, zorder=5)
  ax_span(p16, p84, color="black", ls="-", lw=1.5, alpha=0.3, zorder=4)


## ###############################################################
## STAGE 1 MCMC FITTER
## ###############################################################

class MCMCStage1Routine(base_mcmc.BaseMCMCRoutine):
  def __init__(self, output_directory, x_values, y_values, verbose):
    self.log10_e = numpy.log10(numpy.exp(1))
    self.max_time = numpy.max(x_values)
    super().__init__(
      output_directory    = output_directory,
      routine_name        = "stage1",
      verbose             = verbose,
      x_values            = x_values,
      y_values            = numpy.log10(y_values),
      initial_params      = (-20, 0.85 * numpy.max(x_values), 0.5),
      y_data_label        = r"$\log_{10}(E_{\mathrm{mag}})$",
      fitted_param_labels = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$\gamma$",
        r"$t_{\mathrm{approx}}$",
      ]
    )

  def _model(self, param_vector):
    (log10_init_energy, transition_time, gamma) = param_vector
    ## mask reduced ssd phases
    mask_exp = self.x_values < transition_time
    mask_sat = ~mask_exp
    ## model energy evolution
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

  def _annotate_fitted_params(self, axs):
    tt_samples = self.fitted_posterior_samples[:,1]
    log10_gamma_samples = self.log10_e * numpy.array(self.fitted_posterior_samples[:,2])
    plot_param_percentiles(axs[1], log10_gamma_samples, orientation="horizontal")
    for row_index in range(len(axs)):
      plot_param_percentiles(axs[row_index], tt_samples, orientation="vertical")


## END OF MODULE