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
def _model(time, fit_params, known_params):
  time = numpy.array(time)
  (time_nl_start, time_sat_start) = fit_params
  (max_time, init_energy, sat_energy, gamma, beta) = known_params
  energy = numpy.zeros_like(time)
  ## mask different ssd phases
  mask_exp_phase = time <= time_nl_start
  mask_nl_phase  = (time_nl_start < time) & (time <= time_sat_start)
  mask_sat_phase = time_sat_start < time
  ## compute energy in different ssd phases
  alpha = (sat_energy - init_energy * numpy.exp(gamma * time_nl_start)) / (time_sat_start - time_nl_start)**beta
  energy[mask_exp_phase] = init_energy * numpy.exp(gamma * time[mask_exp_phase])
  energy[mask_nl_phase]  = init_energy * numpy.exp(gamma * time_nl_start) * (1 + time[mask_nl_phase] - time_nl_start) + alpha * (time[mask_nl_phase] - time_nl_start)**beta
  energy[mask_sat_phase] = init_energy * numpy.exp(gamma * time_nl_start) * (1 + time_sat_start - time_nl_start) + alpha * (time_sat_start - time_nl_start)**beta
  return energy

def model(time, fit_params, known_params):
  time = numpy.array(time)
  (time_nl_start, time_sat_start) = fit_params
  (max_time, init_energy, sat_energy, gamma, beta) = known_params
  ## exponential phase
  energy_exp_phase = init_energy * numpy.exp(gamma * time)
  ## polynomial phase
  energy_exp_phase_end = init_energy * numpy.exp(gamma * time_nl_start)
  derivative_exp_phase_end = init_energy * gamma * numpy.exp(gamma * time_nl_start) / beta 
  energy_nl_phase = energy_exp_phase_end + derivative_exp_phase_end * numpy.maximum(time - time_nl_start, 0)**beta
  ## saturated phase
  energy_sat_phase = derivative_exp_phase_end * (time_sat_start - time_nl_start)**beta
  ## calculate smooth transition between phases
  logistic_function_1 = expit(-(time - time_nl_start))
  logistic_function_2 = expit(-(time - time_sat_start))
  weights_exp_phase = logistic_function_1
  weights_nl_phase  = logistic_function_2 - logistic_function_1
  weights_sat_phase = 1 - logistic_function_2
  energy = weights_exp_phase * energy_exp_phase + weights_nl_phase * energy_nl_phase + weights_sat_phase * energy_sat_phase
  return energy

def check_params_are_valid(fit_params, known_params, debug_mode=False):
  (time_nl_start, time_sat_start) = fit_params
  (max_time, init_energy, sat_energy, gamma, beta) = known_params
  errors = []
  if not (10 < time_nl_start < 0.85 * max_time):
    errors.append(f"`time_nl_start` ({time_nl_start}) must be greater than 0 and less than time_sat_start ({time_sat_start}).")
  if not (time_nl_start < time_sat_start < max_time):
    errors.append(f"`time_sat_start` ({time_sat_start}) must be less than `max_time` = {max_time}.")
  if not (0.5 < beta < 5):
    errors.append(f"`beta` ({beta}) must be between 0 and 10.")
  if len(errors) > 0:
    if debug_mode: print("\n".join(errors))
    return False
  return True

def log_likelihood(fit_params, time, measured_energy, known_params, nl_weight=10.0):
  if not check_params_are_valid(fit_params, known_params): return -numpy.inf
  estimated_energy = model(time, fit_params, known_params)
  if not numpy.all(numpy.isfinite(estimated_energy)): return -numpy.inf
  time_nl_start, time_sat_start, beta = fit_params
  mask_nl_phase = (time > time_nl_start) & (time <= time_sat_start)
  weights = numpy.ones_like(time, dtype=float)
  weights[mask_nl_phase] = nl_weight
  estimated_energy = numpy.clip(estimated_energy, 1e-12, None)
  measured_energy = numpy.clip(measured_energy, 1e-12, None)
  if not numpy.all(numpy.isfinite(estimated_energy)): return -numpy.inf
  if numpy.any(estimated_energy <= 0): return -numpy.inf
  log_diff = numpy.log10(measured_energy) - numpy.log10(estimated_energy)
  return -0.5 * numpy.sum(weights * numpy.square(log_diff))

def log_prior(fit_params, known_params):
  if not check_params_are_valid(fit_params, known_params): return -numpy.inf
  return 0 # uniform prior

def log_posterior(fit_params, time, measured_energy, known_params):
  log_prior_value = log_prior(fit_params, known_params)
  if not numpy.isfinite(log_prior_value): return -numpy.inf
  log_likelihood_value = log_likelihood(fit_params, time, measured_energy, known_params)
  return log_prior_value + log_likelihood_value

def estimate_params_with_mcmc(time, measured_energy, initial_guess, known_params):
  num_walkers   = 200
  num_steps     = 5000
  burn_in_steps = 2000
  num_params    = len(initial_guess)
  param_positions = numpy.array(initial_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
  sampler = emcee.EnsembleSampler(num_walkers, num_params, log_posterior, args=(time, measured_energy, known_params))
  sampler.run_mcmc(param_positions, num_steps)
  samples = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
  save_corner_plot(samples)
  plot_chain_evolution(sampler)
  return numpy.median(samples, axis=0)

def save_corner_plot(samples):
  fig = corner.corner(samples)
  plot_manager.save_figure(fig, "mcmc_corner_plot.png")

def plot_chain_evolution(sampler):
  chain = sampler.get_chain()
  _, num_walkers, num_params = chain.shape
  fig, axs = plot_manager.create_figure(num_rows=num_params, axis_shape=(6, 10), share_x=True)
  for param_index in range(num_params):
    ax = axs[param_index]
    for walker_index in range(num_walkers):
      ax.plot(chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
  axs[-1].set_xlabel("steps")
  plot_manager.save_figure(fig, "mcmc_chain_evolution.png")


## ###############################################################
## ESTIMATE TRANSITION
## ###############################################################
def main():
  sanity_check = 1
  run_mcmc = 1
  fig, axs = plot_manager.create_figure(
    num_rows   = 3,
    share_x    = True,
    axis_shape = (6, 10)
  )
  time_start = 20.0
  time, measured_energy = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re500/Mach0.3/Pm1/576",
    dataset_name = "mag",
    time_start   = time_start,
  )
  time = time - time_start # skip straight to the exponential phase
  ## uniformly smaple the data
  data_plot_args = dict(color="blue", marker="o", ms=5, ls="", zorder=3, label="measured values")
  num_interp_points = 200
  _interp_time = numpy.linspace(numpy.min(time), numpy.max(time), num_interp_points)
  interp_time, log10_interp_energy = interpolate_data.interpolate_1d(time, measured_energy, _interp_time, kind="linear")
  time = interp_time
  measured_energy = log10_interp_energy
  ## get constraining knowns
  max_time        = numpy.max(time)
  num_points      = len(measured_energy)
  exp_end_index   = int(0.25 * num_points)
  sat_start_index = int(0.75 * num_points)
  init_energy     = (measured_energy[0])
  sat_energy      = numpy.median(measured_energy[sat_start_index:])
  gamma           = numpy.median(numpy.gradient(numpy.log(measured_energy[:exp_end_index]), time[:exp_end_index]))
  known_params    = (max_time, init_energy, sat_energy, gamma)
  linearised_exp_data = numpy.log10(init_energy) + gamma * time * numpy.log10(numpy.exp(1))
  plot_data.plot_wo_scaling_axis(axs[1], time, linearised_exp_data, lw=1.5)
  ## sanity check
  if sanity_check:
    my_plot_args = dict(color="green", marker="o", ms=5, ls="-", lw=2, zorder=3, label="my estimate")
    my_params = [100, 150, 1.7]
    axs[0].axvline(x=my_params[0], color="green", ls="--")
    axs[0].axvline(x=my_params[1], color="green", ls="--")
    if not check_params_are_valid(my_params, known_params, debug_mode=True):
      raise ValueError("Error: my estimated paramaters are invalid!")
    my_estimated_energy = model(time, my_params, known_params)
    my_residual = numpy.log10(measured_energy) - numpy.log10(my_estimated_energy)
  ## run the mcmc routine
  if run_mcmc:
    mcmc_plot_args = dict(color="red", marker="o", ms=2, ls="-", lw=2, zorder=3, label="MCMC estimate")
    initial_guess = ( 0.25*max_time, 0.75*max_time, 1.5 ) # guess
    mcmc_params = estimate_params_with_mcmc(time, measured_energy, initial_guess, known_params)
    print([
      f"{param:.3f}"
      for param in mcmc_params
    ])
    mcmc_estimated_energy = model(time, mcmc_params, known_params)
    mcmc_residual = numpy.log10(measured_energy) - numpy.log10(mcmc_estimated_energy)
  ## plot data
  axs[0].plot(time, measured_energy, **data_plot_args)
  axs[1].plot(time, numpy.log10(measured_energy), **data_plot_args)
  if sanity_check:
    axs[0].plot(time, my_estimated_energy, **my_plot_args)
    axs[1].plot(time, numpy.log10(my_estimated_energy), **my_plot_args)
    axs[2].plot(time, my_residual, **my_plot_args)
    axs[2].axhline(y=numpy.median(my_residual), **my_plot_args)
  if run_mcmc:
    axs[0].plot(time, mcmc_estimated_energy, **mcmc_plot_args)
    axs[1].plot(time, numpy.log10(mcmc_estimated_energy), **mcmc_plot_args)
    axs[2].plot(time, mcmc_residual, **mcmc_plot_args)
    axs[2].axhline(y=numpy.median(mcmc_residual), **mcmc_plot_args)
    axs[0].axvline(x=mcmc_params[0], color="red", ls="--")
    axs[0].axvline(x=mcmc_params[1], color="red", ls="--")
  axs[1].axhline(y=numpy.log10(init_energy), color="black", ls="--")
  axs[1].axhline(y=numpy.log10(sat_energy), color="black", ls="--")
  axs[1].axvline(x=time[exp_end_index], color="black", ls="--")
  axs[1].axvline(x=time[sat_start_index], color="black", ls="--")
  axs[2].axhline(y=0, color="black", ls="--")
  ## label plots
  axs[0].legend(loc="lower right")
  axs[0].set_ylabel("energy")
  axs[1].set_ylabel("$\log_{10}$(energy)")
  axs[2].set_ylabel("residual of $\log_{10}$(energies)")
  axs[-1].set_xlabel("time")
  plot_manager.save_figure(fig, "estimate_using_mcmc.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT