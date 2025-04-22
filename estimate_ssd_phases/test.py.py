## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
from scipy.special import expit
from jormi.ww_io import flash_data
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager, plot_data


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def check_params_are_valid(fit_params, known_params):
  time_nl_start, time_sat_start = fit_params
  max_time, init_energy, sat_energy, gamma, beta = known_params
  errors = []
  if not (10 < time_nl_start < 0.85 * max_time):
    errors.append(f"`time_nl_start` ({time_nl_start}) must be >10 and <85% of max_time.")
  if not (time_nl_start < time_sat_start < max_time):
    errors.append(f"`time_sat_start` ({time_sat_start}) must be > time_nl_start and < max_time.")
  if len(errors) > 0:
    print("\n".join(errors))
    return False
  return True

def load_data():
  time_start = 20.0
  time, measured_energy = flash_data.read_vi_data(
    directory="/scratch/jh2/nk7952/Re500/Mach0.3/Pm1/576",
    dataset_name="mag",
    time_start=time_start,
  )
  time = time - time_start
  num_interp_points = 200
  _interp_time = numpy.linspace(numpy.min(time), numpy.max(time), num_interp_points)
  interp_time, log10_interp_energy = interpolate_data.interpolate_1d(time, measured_energy, _interp_time, kind="linear")
  return interp_time, log10_interp_energy


## ###############################################################
## MCMC OPERATOR
## ###############################################################
class MCMCModel:
  def __init__(self, time, measured_energy, known_params):
    self.time = time
    self.measured_energy = measured_energy
    self.known_params = known_params

  def model(self, fit_params):
    time = numpy.array(self.time)
    time_nl_start, time_sat_start = fit_params
    max_time, init_energy, sat_energy, gamma, beta = self.known_params

    # Exponential phase
    energy_exp_phase = init_energy * numpy.exp(gamma * time)
    energy_exp_phase_end = init_energy * numpy.exp(gamma * time_nl_start)
    derivative_exp_phase_end = init_energy * gamma * numpy.exp(gamma * time_nl_start) / beta
    energy_nl_phase = energy_exp_phase_end + derivative_exp_phase_end * numpy.maximum(time - time_nl_start, 0)**beta

    # Saturated phase
    energy_sat_phase = derivative_exp_phase_end * (time_sat_start - time_nl_start)**beta

    # Logistic function for smooth transitions
    logistic_function_1 = expit(-(time - time_nl_start))
    logistic_function_2 = expit(-(time - time_sat_start))
    weights_exp_phase = logistic_function_1
    weights_nl_phase = logistic_function_2 - logistic_function_1
    weights_sat_phase = 1 - logistic_function_2

    # Combine phases
    energy = weights_exp_phase * energy_exp_phase + weights_nl_phase * energy_nl_phase + weights_sat_phase * energy_sat_phase
    return energy

  def log_likelihood(self, fit_params, nl_weight=10.0):
    if not check_params_are_valid(fit_params, self.known_params):
      return -numpy.inf
    estimated_energy = self.model(fit_params)
    if not numpy.all(numpy.isfinite(estimated_energy)):
      return -numpy.inf

    # Calculate residuals with weights for nonlinear phase
    time_nl_start, time_sat_start = fit_params
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
    lp = self.log_prior(fit_params)
    if not numpy.isfinite(lp):
      return -numpy.inf
    ll = self.log_likelihood(fit_params)
    return lp + ll

  def estimate_params_with_mcmc(self, initial_guess, num_walkers=200, num_steps=5000, burn_in_steps=2000):
    num_params = len(initial_guess)
    param_positions = numpy.array(initial_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
    sampler = emcee.EnsembleSampler(num_walkers, num_params, self.log_posterior)
    sampler.run_mcmc(param_positions, num_steps)
    samples = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    return numpy.median(samples, axis=0)


## ###############################################################
## PLOT MANAGER
## ###############################################################
class PlotManager:
  def __init__(self, fig_size=(10, 6)):
    self.fig_size = fig_size

  def create_figure(self, num_rows=1, num_cols=1, **kwargs):
    fig, axs = plot_manager.create_figure(num_rows=num_rows, num_cols=num_cols, fig_size=self.fig_size, **kwargs)
    return fig, axs

  def save_figure(self, fig, filename):
    plot_manager.save_figure(fig, filename)

  def plot_mcmc_chain_evolution(self, sampler, tag):
    chain = sampler.get_chain()
    _, num_walkers, num_params = chain.shape
    fig, axs = self.create_figure(num_params=num_params, num_rows=num_params, share_x=True)
    for param_index in range(num_params):
      ax = axs[param_index]
      for walker_index in range(num_walkers):
        ax.plot(chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
    axs[-1].set_xlabel("steps")
    self.save_figure(fig, f"mcmc_chain_evolution_beta{tag}.png")

  def plot_corner(self, samples, tag=""):
    fig = corner.corner(samples)
    self.save_figure(fig, f"mcmc_corner_plot_beta{tag}.png")


def main():
  # Load and interpolate data
  time, measured_energy = load_data()

  # Parameters for known constants
  max_time = numpy.max(time)
  num_points = len(measured_energy)
  exp_end_index = int(0.25 * num_points)
  sat_start_index = int(0.75 * num_points)
  init_energy = measured_energy[0]
  sat_energy = numpy.median(measured_energy[sat_start_index:])
  gamma = numpy.median(numpy.gradient(numpy.log(measured_energy[:exp_end_index]), time[:exp_end_index]))

  # Initialize MCMCModel with known parameters
  known_params = (max_time, init_energy, sat_energy, gamma, 1.7)
  mcmc_model = MCMCModel(time, measured_energy, known_params)

  # Perform MCMC to estimate parameters
  initial_guess = [0.25 * max_time, 0.75 * max_time]
  estimated_params = mcmc_model.estimate_params_with_mcmc(initial_guess)

  # Plotting results
  plot_manager = PlotManager()

  fig, axs = plot_manager.create_figure(num_rows=3, share_x=True, axis_shape=(6, 10))

  # Plot data and model predictions
  axs[0].plot(time, measured_energy, color="blue", label="measured")
  axs[1].plot(time, numpy.log10(measured_energy), color="blue", label="log(measured)")

  # Add model results to plots
  mcmc_estimated_energy = mcmc_model.model(estimated_params)
  axs[0].plot(time, mcmc_estimated_energy, color="red", label="MCMC fit")

  # Plot residuals
  residuals = numpy.log10(measured_energy) - numpy.log10(mcmc_estimated_energy)
  axs[2].plot(time, residuals, color="red", label="residuals")

  plot_manager.save_figure(fig, "mcmc_results.png")

if __name__ == "__main__":
  main()
