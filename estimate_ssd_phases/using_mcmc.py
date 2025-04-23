## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
from scipy.special import expit
from jormi.ww_io import flash_data
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def check_params_are_valid(fit_params, known_params, debug_mode=False):
  time_nl_start, time_sat_start = fit_params
  max_time, init_energy, sat_energy, gamma, beta = known_params
  errors = []
  if not (10 < time_nl_start < 0.85 * max_time):
    errors.append(f"`time_nl_start` ({time_nl_start}) must be >10 and <85% of max_time.")
  if not (time_nl_start < time_sat_start < max_time):
    errors.append(f"`time_sat_start` ({time_sat_start}) must be > time_nl_start and < max_time.")
  if len(errors) > 0:
    if debug_mode: print("\n".join(errors))
    return False
  return True

def load_data(start_time = 20.0):
  time, measured_energy = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re500/Mach0.3/Pm1/576",
    dataset_name = "mag",
    start_time   = start_time,
  )
  time = time - start_time
  interp_time, interp_energy = interpolate_data.interpolate_1d(
    x_values = time,
    y_values = measured_energy,
    x_interp = numpy.linspace(numpy.min(time), numpy.max(time), 200),
    kind     = "linear"
  )
  return interp_time, interp_energy


## ###############################################################
## MCMC OPERATOR
## ###############################################################
class MCMCModel:
  def __init__(self, time, measured_energy, known_params):
    self.time = time
    self.measured_energy = measured_energy
    self.known_params = known_params

  def model1(self, fit_params):
    time = numpy.array(self.time)
    (time_nl_start, time_sat_start) = fit_params
    (max_time, init_energy, sat_energy, gamma, beta) = self.known_params
    ## exponential phase
    energy_exp_phase = init_energy * numpy.exp(gamma * time)
    energy_exp_phase_end = init_energy * numpy.exp(gamma * time_nl_start)
    derivative_exp_phase_end = init_energy * gamma * numpy.exp(gamma * time_nl_start) / beta
    energy_nl_phase = energy_exp_phase_end + derivative_exp_phase_end * numpy.maximum(time - time_nl_start, 0)**beta
    ## saturated phase
    energy_sat_phase = derivative_exp_phase_end * (time_sat_start - time_nl_start)**beta
    ## logistic sigmoid function for smooth transitions
    f1 = expit(-(time - time_nl_start))
    f2 = expit(-(time - time_sat_start))
    weights_exp_phase = f1
    weights_nl_phase  = f2 - f1
    weights_sat_phase = 1 - f2
    ## combine phases
    energy = weights_exp_phase * energy_exp_phase + weights_nl_phase * energy_nl_phase + weights_sat_phase * energy_sat_phase
    return energy

  def model2(self, fit_params):
    time = numpy.array(time)
    (time_nl_start, time_sat_start) = fit_params
    (max_time, init_energy, sat_energy, gamma, beta) = self.known_params
    ## mask different ssd phases
    mask_exp_phase = time <= time_nl_start
    mask_nl_phase  = (time_nl_start < time) & (time <= time_sat_start)
    mask_sat_phase = time_sat_start < time
    ## compute energy in different ssd phases
    alpha = (sat_energy - init_energy * numpy.exp(gamma * time_nl_start)) / (time_sat_start - time_nl_start)**beta
    energy = numpy.zeros_like(time)
    energy[mask_exp_phase] = init_energy * numpy.exp(gamma * time[mask_exp_phase])
    energy[mask_nl_phase]  = init_energy * numpy.exp(gamma * time_nl_start) * (1 + time[mask_nl_phase] - time_nl_start) + alpha * (time[mask_nl_phase] - time_nl_start)**beta
    energy[mask_sat_phase] = init_energy * numpy.exp(gamma * time_nl_start) * (1 + time_sat_start - time_nl_start) + alpha * (time_sat_start - time_nl_start)**beta
    return energy

  def log_likelihood(self, fit_params, nl_weight=10.0):
    if not check_params_are_valid(fit_params, self.known_params):
      return -numpy.inf
    estimated_energy = self.model1(fit_params)
    if not numpy.all(numpy.isfinite(estimated_energy)):
      return -numpy.inf
    (time_nl_start, time_sat_start) = fit_params
    mask_nl_phase = (self.time > time_nl_start) & (self.time <= time_sat_start)
    weights = numpy.ones_like(self.time, dtype=float)
    weights[mask_nl_phase] = nl_weight
    estimated_energy = numpy.clip(estimated_energy, 1e-12, None)
    measured_energy = numpy.clip(self.measured_energy, 1e-12, None)
    log_diff = numpy.log10(measured_energy) - numpy.log10(estimated_energy)
    return -0.5 * numpy.sum(weights * numpy.square(log_diff))

  def log_prior(self, fit_params):
    if not check_params_are_valid(fit_params, self.known_params):
      return -numpy.inf
    return 0
  
  def log_posterior(self, fit_params):
    log_prior_value = self.log_prior(fit_params)
    if not numpy.isfinite(log_prior_value): return -numpy.inf
    log_likelihood_value = self.log_likelihood(fit_params)
    return log_prior_value + log_likelihood_value

  def estimate_params_with_mcmc(
      self,
      initial_guess,
      num_walkers   = 200,
      num_steps     = 5000,
      burn_in_steps = 2000
    ):
    num_params = len(initial_guess)
    param_positions = numpy.array(initial_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
    sampler = emcee.EnsembleSampler(num_walkers, num_params, self.log_posterior)
    sampler.run_mcmc(param_positions, num_steps)
    samples = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    return numpy.median(samples, axis=0), sampler, samples


## ###############################################################
## PLOT MANAGER
## ###############################################################
class PlotManager:
  def __init__(self, axis_shape=(4, 6)):
    self.axis_shape = axis_shape

  def create_figure(self, num_rows=1, num_cols=1, **kwargs):
    fig, axs = plot_manager.create_figure(
      num_rows   = num_rows,
      num_cols   = num_cols,
      axis_shape = self.axis_shape,
      **kwargs
    )
    return fig, axs

  def save_figure(self, fig, filename):
    plot_manager.save_figure(fig, filename)

  def plot_mcmc_chain_evolution(self, sampler, beta):
    chain = sampler.get_chain()
    _, num_walkers, num_params = chain.shape
    fig, axs = self.create_figure(num_rows=num_params, share_x=True)
    for param_index in range(num_params):
      ax = axs[param_index]
      for walker_index in range(num_walkers):
        ax.plot(chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
    axs[-1].set_xlabel("steps")
    self.save_figure(fig, f"mcmc_chain_evolution_beta={beta}.png")

  def plot_corner(self, samples, beta):
    fig = corner.corner(samples)
    self.save_figure(fig, f"mcmc_corner_plot_beta={beta}.png")


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  ## load and interpolate data
  time, measured_energy = load_data()
  ## parameters for known constants
  max_time        = numpy.max(time)
  init_energy     = measured_energy[0]
  num_points      = len(measured_energy)
  sat_start_index = int(0.75 * num_points)
  exp_end_index   = int(0.25 * num_points)
  sat_energy      = numpy.median(measured_energy[sat_start_index:])
  gamma           = numpy.median(numpy.gradient(numpy.log(measured_energy[:exp_end_index]), time[:exp_end_index]))
  ## estimate parameters using MCMC
  for beta in [1.0, 2.0]:
    known_params = (max_time, init_energy, sat_energy, gamma, beta)
    mcmc_model = MCMCModel(time, measured_energy, known_params)
    initial_guess = [0.25 * max_time, 0.75 * max_time]
    estimated_params, sampler, samples = mcmc_model.estimate_params_with_mcmc(initial_guess)
    ## MCMC diagnostic plots
    plot_manager = PlotManager()
    plot_manager.plot_mcmc_chain_evolution(sampler, beta)
    plot_manager.plot_corner(samples, beta)
    ## model vs data
    fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
    axs[0].plot(time, measured_energy, color="blue", label="measured")
    axs[1].plot(time, numpy.log10(measured_energy), color="blue")
    mcmc_estimated_energy = mcmc_model.model1(estimated_params)
    axs[0].plot(time, mcmc_estimated_energy, color="red", label="MCMC fit")
    axs[1].plot(time, numpy.log10(mcmc_estimated_energy), color="red")
    for ax in list(axs):
      ax.axvline(x=estimated_params[0], color="black", ls="--")
      ax.axvline(x=estimated_params[1], color="black", ls="--")
    residuals = numpy.log10(measured_energy) - numpy.log10(mcmc_estimated_energy)
    axs[2].plot(time, residuals, color="red", label="residuals")
    axs[2].axhline(y=0, color="black", ls="--")
    axs[0].set_ylabel(r"$E_{\rm mag}$")
    axs[1].set_ylabel(r"$\log_{10}(E_{\rm mag})$")
    axs[2].set_ylabel(r"${\rm d} \log_{10}(E_{\rm mag}) / {\rm d} t$")
    axs[-1].set_xlabel("t")
    plot_manager.save_figure(fig, f"mcmc_results_beta={beta}.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT