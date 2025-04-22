## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
import random
from jormi.ww_io import flash_data, directory_manager
from jormi.ww_plots import plot_manager
import utils


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def model(time, params):
  init_energy, gamma, transition_time = params
  return numpy.where(
    time < transition_time,
    init_energy + gamma * time,
    init_energy + gamma * transition_time
  )

def check_params_are_valid(params):
  init_energy, gamma, transition_time = params
  if not (-100 < init_energy < 0): return False
  if not (0 < gamma < 1): return False
  if transition_time <= 0 or 1e3 <= transition_time: return False
  return True

def log_likelihood(params, time, measured_energy):
  if not check_params_are_valid(params): return -numpy.inf
  estimated_energy = model(time, params)
  return -0.5 * numpy.sum(numpy.square(measured_energy - estimated_energy))

def log_prior(params):
  if not check_params_are_valid(params): return -numpy.inf
  return 0 # uniform prior

def log_posterior(params, time, measured_energy):
  log_prior_value = log_prior(params)
  if not numpy.isfinite(log_prior_value): return -numpy.inf
  log_likelihood_value = log_likelihood(params, time, measured_energy)
  return log_prior_value + log_likelihood_value

def estimate_params(time, measured_energy, init_params, fig_directory):
  num_walkers     = 100
  num_steps       = 5000
  burn_in_steps   = 1000
  num_params      = len(init_params)
  param_positions = numpy.array(init_params) + 1e-4 * numpy.random.randn(num_walkers, num_params)
  sampler = emcee.EnsembleSampler(num_walkers, num_params, log_posterior, args=(time, measured_energy))
  sampler.run_mcmc(param_positions, num_steps)
  samples = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
  init_energy, growth_rate, transition_time = numpy.median(samples, axis=0)
  save_corner_plot(samples, fig_directory)
  plot_chain_evolution(sampler, fig_directory)
  return [ init_energy, growth_rate, transition_time ]

def save_corner_plot(samples, fig_directory):
  fig = corner.corner(samples)
  plot_manager.save_figure(fig, f"{fig_directory}/dev_mcmc_corner_plot.png")

def plot_chain_evolution(sampler, fig_directory):
  chain = sampler.get_chain()
  _, num_walkers, num_params = chain.shape
  fig, axs = plot_manager.create_figure(num_rows=num_params, axis_shape=(6, 10), share_x=True)
  for param_index in range(num_params):
    ax = axs[param_index]
    for walker_index in range(num_walkers):
      ax.plot(chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
  axs[-1].set_xlabel("steps")
  plot_manager.save_figure(fig, f"{fig_directory}/dev_mcmc_chain_evolution.png")

def generate_data(num_points, time_bounds, init_energy, growth_rate, transition_time):
  time = utils.generate_uniform_domain(
    domain_bounds = time_bounds,
    num_points    = num_points,
  )
  measured_energy = utils.generate_data(
    x_data       = time,
    noise_level  = 3.0,
    init_value   = init_energy,
    growth_rate  = growth_rate,
    x_transition = transition_time,
  )
  return time, measured_energy

def load_data():
  time_start = 20.0
  time, measured_energy = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re500/Mach0.3/Pm1/576",
    dataset_name = "mag",
    time_start   = time_start,
  )
  time = time - time_start
  measured_energy = numpy.log10(measured_energy)
  return time, measured_energy


## ###############################################################
## ESTIMATE TRANSITION
## ###############################################################
def main():
  fig, ax = plot_manager.create_figure(axis_shape=(6, 10))
  fig_directory = "dev_plots"
  directory_manager.init_directory(fig_directory)
  init_energy     = -13
  growth_rate     = 9/100
  transition_time = 120
  true_params = [ init_energy, growth_rate, transition_time ]
  # time, measured_energy = generate_data(100, [0, 500], init_energy, growth_rate, transition_time)
  time, measured_energy = load_data()
  init_params = [-20, 0.5, 50]
  mcmc_params = estimate_params(time, measured_energy, init_params, fig_directory)
  print([
    f"{param:.3f}"
    for param in mcmc_params
  ])
  estimated_energy = model(time, mcmc_params)
  my_estimated_energy = model(time, true_params)
  ax.plot(time, measured_energy, color="blue", marker="o", ms=5, ls="", zorder=3, label="measured values")
  ax.plot(time, estimated_energy, color="red", ls="-", lw=2, zorder=3, label="MCMC estimated model")
  ax.plot(time, my_estimated_energy, color="green", ls="-", lw=2, zorder=3, label="my estimated model")
  ax.set_xlabel("time")
  ax.set_ylabel("energy")
  ax.legend(loc="lower right")
  plot_manager.save_figure(fig, f"{fig_directory}/dev_estimate_using_mcmc.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT