## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
from scipy.ndimage import gaussian_filter1d as scipy_filter1d
from jormi.utils import list_utils
from jormi.ww_io import flash_data
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
# def check_stage2_params_are_valid(fit_params, known_params, print_errors=False):
#   (start_nl_time, start_sat_time) = fit_params
#   (max_time, gamma, sat_energy) = known_params
#   errors = []
#   if not (0.1 * max_time < start_nl_time < 0.85 * max_time):
#     errors.append(f"`start_nl_time` ({start_nl_time}) must be beteeen 10 and 85 percent of `max_time`.")
#   if not (start_nl_time < start_sat_time < max_time):
#     errors.append(f"`start_sat_time` ({start_sat_time}) must be between `start_nl_time` and `max_time`.")
#   if not (100 * (start_sat_time - start_nl_time) / max_time > 5):
#     errors.append(f"non-linear phase duration should be at least 5 percent of the total elapsed time.")
#   if len(errors) > 0:
#     if print_errors: print("\n".join(errors))
#     return False
#   return True

def load_data(num_samples = 100):
  time, measured_energy = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re500/Mach0.3/Pm1/576",
    dataset_name = "mag"
  )
  interp_time, interp_energy = interpolate_data.interpolate_1d(
    x_values = time[1:],
    y_values = measured_energy[1:],
    x_interp = numpy.linspace(time[1], time[-1], num_samples),
    kind     = "linear"
  )
  return interp_time[3:], interp_energy[3:]


## ###############################################################
## STAGE 1 MCMC FITTER
## ###############################################################
class Stage1:
  def __init__(self, time, measured_energy):
    self.time = time
    self.measured_log10_energy = numpy.log10(measured_energy)
    self.max_time = numpy.max(self.time)
    self.param_labels = [
      r"$\log_{10}(E_{\mathrm{init}})$",
      r"$t_{\mathrm{approx}}$",
      r"$\gamma$"
    ]

  def estimate_params(
      self,
      initial_guess,
      num_walkers   = 200,
      num_steps     = 5000,
      burn_in_steps = 2000,
      skip_mcmc     = False,
    ):
    if not self._check_params_are_valid(initial_guess, print_errors=True):
      raise ValueError("Inital guess is invalid!")
    if skip_mcmc:
      self._plot_model_results(initial_guess)
      return
    num_params = len(initial_guess)
    param_positions = numpy.array(initial_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
    sampler = emcee.EnsembleSampler(num_walkers, num_params, self._log_posterior)
    sampler.run_mcmc(param_positions, num_steps)
    chain      = sampler.get_chain()
    samples    = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    fit_params = numpy.median(samples, axis=0)
    self._plot_chain_evolution(chain)
    self._corner_plot(samples)
    self._plot_model_results(fit_params)
    return fit_params

  def _log10_energy_model(self, fit_params):
    (log10_init_energy, transition_time, gamma) = fit_params
    mask_exp = self.time < transition_time
    mask_sat = ~mask_exp
    log10_energy = numpy.zeros_like(self.time)
    log10_energy[mask_exp] = log10_init_energy + numpy.log10(numpy.exp(1)) * gamma * self.time[mask_exp]
    log10_energy[mask_sat] = log10_init_energy + numpy.log10(numpy.exp(1)) * gamma * transition_time
    return log10_energy

  def _check_params_are_valid(self, fit_params, print_errors=False):
    (log10_init_energy, transition_time, gamma) = fit_params
    errors = []
    if not (-30 < log10_init_energy < -5):
      errors.append(f"`log10_init_energy` ({log10_init_energy:.2f}) must be beteeen -20 and -5.")
    if not (0.25 * self.max_time < transition_time < 0.9 * self.max_time):
      errors.append(f"`transition_time` ({transition_time:.2f}) must be beteeen 25 and 90 percent of `max_time` ({self.max_time:.2f}).")
    if not (0 < gamma < 1):
      errors.append(f"`gamma` ({gamma:.2f}) must be between 0 and 1.")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True

  def _log_prior(self, fit_params):
    if not self._check_params_are_valid(fit_params):
      return -numpy.inf
    return 0

  def _log_likelihood(self, fit_params):
    if not self._check_params_are_valid(fit_params):
        return -numpy.inf
    modelled_log10_energy = self._log10_energy_model(fit_params)
    residual_log10_energy = self.measured_log10_energy - modelled_log10_energy
    try:
        log_likelihood = -0.5 * numpy.sum(numpy.square(residual_log10_energy))
        if not numpy.isfinite(log_likelihood):
            return -numpy.inf
        return log_likelihood
    except Exception as e:
        print("Error in likelihood:", e, fit_params)
        return -numpy.inf

  def _log_posterior(self, fit_params):
    log_prior_value = self._log_prior(fit_params)
    if not numpy.isfinite(log_prior_value): return -numpy.inf
    log_likelihood_value = self._log_likelihood(fit_params)
    return log_prior_value + log_likelihood_value

  def _plot_chain_evolution(self, chain):
    _, num_walkers, num_params = chain.shape
    fig, axs = plot_manager.create_figure(
      num_rows = num_params,
      num_cols = 1,
      share_x  = True
    )
    for param_index in range(num_params):
      ax = axs[param_index]
      for walker_index in range(num_walkers):
        ax.plot(chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
      ax.set_ylabel(self.param_labels[param_index])
    axs[-1].set_xlabel("steps")
    plot_manager.save_figure(fig, "mcmc_stage_1_chain_evolution.png")

  def _corner_plot(self, samples):
    fig = corner.corner(samples, labels=self.param_labels)
    plot_manager.save_figure(fig, "mcmc_stage_1_corner_plot.png")

  def _plot_model_results(self, fit_params):
    fig, axs = plot_manager.create_figure(num_rows=2, share_x=True)
    data_args = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    axs[0].plot(self.time, self.measured_log10_energy, **data_args)
    model_args = dict(color="red", ls="-", lw=1.5, zorder=5)
    modelled_log10_energy = self._log10_energy_model(fit_params)
    residuals = self.measured_log10_energy - modelled_log10_energy
    axs[0].plot(self.time, modelled_log10_energy, **model_args)
    axs[1].plot(self.time, residuals, **model_args)
    axs[1].axhline(y=0.0, color="black", ls="--")
    axs[0].set_ylabel(r"$\log_{10}(E_{\rm mag})$")
    axs[1].set_ylabel(r"relative error")
    axs[1].set_xlabel("t")
    plot_manager.save_figure(fig, f"mcmc_stage_1_fit.png")


# ## ###############################################################
# ## STAGE 2 MCMC FITTER
# ## ###############################################################
# class Stage2:
#   def __init__(self, time, measured_energy, dlny_dt, dy_dt, known_params):
#     ln_energy = numpy.log(measured_energy)
#     dlny_dt   = numpy.gradient(ln_energy, time)
#     dlny_dt   = scipy_filter1d(dlny_dt, 2.0)
#     dy_dt     = numpy.gradient(measured_energy, time)
#     dy_dt     = scipy_filter1d(dy_dt, 2.0)
#     self.time            = time
#     self.measured_energy = measured_energy
#     self.dlny_dt         = dlny_dt
#     self.dy_dt           = dy_dt
#     self.known_params    = known_params

#   def _dlny_dt_model(time, fit_params, known_params):
#     time = numpy.array(time)
#     (start_nl_time, start_sat_time) = fit_params
#     (max_time, gamma, sat_energy) = known_params
#     ## mask different ssd phases
#     mask_exp_phase = time <= start_nl_time
#     mask_nl_phase  = (start_nl_time < time) & (time <= start_sat_time)
#     ## model
#     slope   = gamma / (start_sat_time - start_nl_time)
#     dlny_dt = numpy.zeros_like(time)
#     dlny_dt[mask_exp_phase] = gamma
#     dlny_dt[mask_nl_phase]  = gamma - slope * (time[mask_nl_phase] - start_nl_time)
#     return dlny_dt

#   def _energy_model(time, fit_params, known_params):
#     time = numpy.array(time)
#     (start_nl_time, start_sat_time) = fit_params
#     (max_time, gamma, sat_energy) = known_params
#     ## mask different ssd phases
#     mask_nl_phase  = (start_nl_time < time) & (time <= start_sat_time)
#     mask_sat_phase = start_sat_time < time
#     ## model
#     slope  = sat_energy / (start_sat_time - start_nl_time)
#     energy = numpy.zeros_like(time)
#     energy[mask_nl_phase]  = slope * (time[mask_nl_phase] - start_nl_time)
#     energy[mask_sat_phase] = sat_energy
#     return energy

#   def estimate_params(
#       self,
#       initial_guess,
#       num_walkers   = 200,
#       num_steps     = 5000,
#       burn_in_steps = 2000,
#       skip_mcmc     = False,
#     ):
#     if skip_mcmc:
#       self._plot_model_results(initial_guess)
#       return
#     num_params = len(initial_guess)
#     param_positions = numpy.array(initial_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
#     sampler = emcee.EnsembleSampler(num_walkers, num_params, self._log_posterior)
#     sampler.run_mcmc(param_positions, num_steps)
#     chain            = sampler.get_chain()
#     samples          = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
#     fit_params = numpy.median(samples, axis=0)
#     self._plot_chain_evolution(chain)
#     self._corner_plot(samples)
#     self._plot_model_results(fit_params)

#   def _log_likelihood(self, fit_params):
#     if not check_params_are_valid(fit_params, self.known_params):
#       return -numpy.inf
#     (_, start_sat_time) = fit_params
#     start_sat_index = list_utils.get_index_of_closest_value(self.time, start_sat_time)
#     modelled_dlny_dt = _dlny_dt_model(self.time, fit_params, self.known_params)
#     residual_dlny_dt = (self.dlny_dt - modelled_dlny_dt) / self.dlny_dt
#     try:
#       return -0.5 * numpy.sum(numpy.square(residual_dlny_dt[:start_sat_index]))
#     except Exception as e:
#       print("Error in likelihood:", e, fit_params)
#       return -numpy.inf

#   def _log_prior(self, fit_params):
#     if not check_params_are_valid(fit_params, self.known_params):
#       return -numpy.inf
#     return 0
  
#   def _log_posterior(self, fit_params):
#     log_prior_value = self._log_prior(fit_params)
#     if not numpy.isfinite(log_prior_value): return -numpy.inf
#     log_likelihood_value = self._log_likelihood(fit_params)
#     return log_prior_value + log_likelihood_value

#   def _plot_chain_evolution(self, chain):
#     _, num_walkers, num_params = chain.shape
#     fig, axs = plot_manager.create_figure(
#       num_rows = num_params,
#       num_cols = 1,
#       share_x  = True
#     )
#     for param_index in range(num_params):
#       ax = axs[param_index]
#       for walker_index in range(num_walkers):
#         ax.plot(chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
#     axs[-1].set_xlabel("steps")
#     plot_manager.save_figure(fig, "mcmc_stage_2_chain_evolution.png")

#   def _corner_plot(self, samples):
#     fig = corner.corner(samples)
#     plot_manager.save_figure(fig, "mcmc_stage_2_corner_plot.png")

#   def _plot_model_results(self, fit_params):
#     fig, axs = plot_manager.create_figure(
#       num_rows   = 3,
#       num_cols   = 2,
#       share_x    = True,
#       axis_shape = (6, 10),
#       x_spacing  = 0.3,
#       y_spacing  = 0.1,
#     )
#     data_args = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
#     axs[0,0].plot(self.time, numpy.log(self.measured_energy), **data_args)
#     axs[0,1].plot(self.time, self.measured_energy, **data_args)
#     axs[1,0].plot(self.time, self.dlny_dt, **data_args)
#     axs[1,1].plot(self.time, self.dy_dt, **data_args)
#     mcmc_args = dict(color="green", ls="-", lw=2.0, zorder=3)
#     modelled_energy  = _energy_model(fit_params)
#     modelled_dlny_dt = _dlny_dt_model(fit_params)
#     modelled_slope   = compute_linear_slope(fit_params, self.known_params)
#     axs[0,1].plot(self.time, modelled_energy, **mcmc_args)
#     axs[1,0].plot(self.time, modelled_dlny_dt, **mcmc_args)
#     axs[1,1].axhline(y=modelled_slope, **mcmc_args)
#     residuals_dlny_dt = (self.dlny_dt - modelled_dlny_dt) / self.dlny_dt
#     residuals_energy  = (self.measured_energy - modelled_energy) / self.measured_energy
#     axs[2,0].plot(self.time, residuals_dlny_dt, **mcmc_args)
#     axs[2,1].plot(self.time, residuals_energy, **mcmc_args)
#     for row_index in range(3):
#       for col_index in range(2):
#         ax = axs[row_index, col_index]
#         ax.axvline(x=fit_params[0], color="green", ls="--", lw=2.0, zorder=2)
#         ax.axvline(x=fit_params[1], color="green", ls="--", lw=2.0, zorder=2)
#     axs[1,0].axhline(y=0.0, color="red", ls="--", lw=2.0, zorder=1)
#     axs[1,1].axhline(y=0.0, color="red", ls="--", lw=2.0, zorder=1)
#     axs[2,0].axhline(y=0.0, color="red", ls="--", lw=2.0, zorder=1)
#     axs[2,1].axhline(y=0.0, color="red", ls="--", lw=2.0, zorder=1)
#     (start_nl_time, start_sat_time) = fit_params
#     start_nl_index  = list_utils.get_index_of_closest_value(self.time, start_nl_time)
#     start_sat_index = list_utils.get_index_of_closest_value(self.time, start_sat_time)
#     axs[2,0].axvspan(self.time[0], self.time[start_nl_index-1], color="green", alpha=0.25)
#     axs[2,0].axvspan(self.time[start_nl_index], self.time[start_sat_index], color="red", alpha=0.25)
#     axs[2,1].axvspan(self.time[start_nl_index], self.time[start_sat_index], color="red", alpha=0.25)
#     axs[2,1].axvspan(self.time[start_sat_index+1], self.time[-1], color="green", alpha=0.25)
#     axs[0,0].set_ylabel(r"$\ln(E_{\rm mag})$")
#     axs[0,1].set_ylabel(r"$E_{\rm mag}$")
#     axs[1,0].set_ylabel(r"$({\rm d}/{\rm d}t) \ln(E_{\rm mag})$")
#     axs[1,1].set_ylabel(r"$({\rm d}/{\rm d}t) E_{\rm mag}$")
#     axs[2,0].set_ylabel(r"residuals")
#     axs[2,1].set_ylabel(r"residuals")
#     axs[2,0].set_xlabel("t")
#     axs[2,1].set_xlabel("t")
#     plot_manager.save_figure(fig, f"mcmc_stage_2_result.png")


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  ## load and interpolate data
  time, measured_energy = load_data(70)
  ## stage 1 MCMC fitter
  stage1_guess  = (-20, 0.85 * numpy.max(time), 0.5)
  stage1_mcmc   = Stage1(time, measured_energy)
  stage1_params = stage1_mcmc.estimate_params(stage1_guess)
  # (log10_init_energy, transition_time, gamma) = [-14.72148534, 141.5647634, 0.20296737]
  # # ## fit using mcmc routine
  # # initial_guess = (0.25*max_time, 0.75*max_time)
  # # mcmc_model.estimate_params(initial_guess)
  # my_best_guess = (110, 170)
  # mcmc_model.estimate_params(my_best_guess, skip_mcmc=True)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT