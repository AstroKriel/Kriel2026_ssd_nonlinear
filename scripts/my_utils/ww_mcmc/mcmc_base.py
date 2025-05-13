## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
from pathlib import Path

from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from jormi.ww_io import io_manager
from jormi.utils import list_utils
from jormi.ww_data import compute_stats
from jormi.ww_plots import plot_manager



## ###############################################################
## ROUTINE
## ###############################################################
class BaseMCMCModel:
  def __init__(
      self,
      output_directory : str | Path,
      routine_name     : str,
      x_data           : list | numpy.ndarray,
      y_data           : list | numpy.ndarray,
      param_guess      : tuple[float, ...],
      ll_sigma         : float | list | numpy.ndarray = 1.0,
      param_labels     : list[str] = [],
      verbose          : bool = True
    ):
    self.output_directory = output_directory
    self.routine_name     = routine_name
    self.x_data           = numpy.asarray(x_data)
    self.y_data           = numpy.asarray(y_data)
    self.ll_sigma         = ll_sigma
    self.param_guess      = param_guess
    self.param_labels     = param_labels
    self.verbose          = verbose
    self._validate_inputs()
    self.samples = None

  def _model(self, fit_params: tuple[float, ...]):
    raise NotImplementedError()

  def _check_params_are_valid(self, fit_params: tuple[float, ...], print_errors: bool = False):
    raise NotImplementedError()

  def _plot_model_results(self, fit_params: tuple[float, ...]):
    raise NotImplementedError()

  def _validate_inputs(self):
    if not isinstance(self.x_data, (list, numpy.ndarray)):
      raise ValueError(f"`x_data` should be either a list or array of values.")
    if not isinstance(self.y_data, (list, numpy.ndarray)):
      raise ValueError(f"`y_data` should be either a list or array of values.")
    if len(self.x_data) != len(self.y_data):
      raise ValueError(f"`x_data` and `y_data` should be the same length, but got {len(self.x_data)} vs {len(self.y_data)}.")
    if not isinstance(self.ll_sigma, (float, int)):
      raise ValueError(f"`ll_sigma` should be a scalar.")
    self.ll_sigma = float(self.ll_sigma)

  def estimate_params(
      self,
      num_walkers   : int = 200,
      num_steps     : int = 5000,
      burn_in_steps : int = 2000,
      plot_guess    : bool = False,
    ):
    if not self._check_params_are_valid(self.param_guess, print_errors=True):
      raise ValueError("Initial guess is invalid!")
    if plot_guess:
      self._plot_model_results(self.param_guess)
      return self.param_guess
    print("Estimating parameters...")
    num_params = len(self.param_guess)
    param_positions = numpy.array(self.param_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
    sampler = emcee.EnsembleSampler(num_walkers, num_params, self._log_posterior)
    sampler.run_mcmc(param_positions, num_steps)
    self.chain   = sampler.get_chain()
    self.samples = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    self._compute_scaled_kde()
    self._plot_chain_evolution()
    self._plot_param_estimates()
    self._plot_model_results()

  def print_log_likelihood(self, fit_params):
    ll_value = self._log_likelihood(fit_params)
    print(f"params = ({fit_params}) yields log-likelihood = {ll_value:.2e}")

  def _log_prior(self, fit_params):
    if not self._check_params_are_valid(fit_params):
      return -numpy.inf
    return 0

  def _log_likelihood(self, fit_params):
    ## TODO: look into gaussian likelihood modelling
    if not self._check_params_are_valid(fit_params):
      return -numpy.inf
    try:
      residual = self.y_data - self._model(fit_params)
      ll_value = -0.5 * numpy.sum(numpy.square(residual / self.ll_sigma))
      if not numpy.isfinite(ll_value):
        return -numpy.inf
      return ll_value
    except Exception as e:
      print("Error in likelihood:", e, fit_params)
      return -numpy.inf

  def _log_posterior(self, fit_params):
    lp_value = self._log_prior(fit_params)
    if not numpy.isfinite(lp_value): return -numpy.inf
    ll_value = self._log_likelihood(fit_params)
    return lp_value + ll_value

  def _plot_chain_evolution(self):
    _, num_walkers, num_params = self.chain.shape
    fig, axs = plot_manager.create_figure(num_rows=num_params, num_cols=1, share_x=True)
    for param_index in range(num_params):
      for walker_index in range(num_walkers):
        axs[param_index].plot(self.chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
      axs[param_index].set_ylabel(self.param_labels[param_index])
    axs[-1].set_xlabel("steps")
    fig_name = f"{self.routine_name}_chain_evolution.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)

  def _plot_pdf(self, ax, param_index):
    values = self.samples[:, param_index]
    bin_centers, estimated_pdf = compute_stats.estimate_pdf(values=values, num_bins=20)
    ax.step(bin_centers, estimated_pdf, where="mid", lw=2, color="black")
    p16, p50, p84 = numpy.percentile(values, [16, 50, 84])
    label = f"{self.param_labels[param_index]} $= {p50:.2f}_{{-{p50-p16:.2f}}}^{{+{p84-p50:.2f}}}$"
    ax.set_title(label, pad=15)
    if param_index > 0: ax.tick_params(labelleft=False, labelright=True)
    if param_index < self.samples.shape[1]-1: ax.set_xticklabels([])
    threshold_value = 0.05 * numpy.max(estimated_pdf)
    index_lower = list_utils.find_first_crossing(values=estimated_pdf, target=threshold_value, direction="rising")
    index_upper = list_utils.find_first_crossing(values=estimated_pdf, target=threshold_value, direction="falling")
    bin_lower = bin_centers[index_lower]
    bin_upper = bin_centers[index_upper]
    return (bin_lower, bin_upper)

  def _plot_param_estimates(self):
    _, num_params = self.samples.shape
    fig, axs = plot_manager.create_figure(
      num_cols   = num_params,
      num_rows   = num_params,
      axis_shape = (5,5)
    )
    param_mins = []
    param_maxs = []
    for row_param_index in range(num_params):
      for col_param_index in range(num_params):
        ax = axs[row_param_index, col_param_index]
        if row_param_index == col_param_index:
          param_min, param_max = self._plot_pdf(ax, row_param_index)
          param_mins.append(param_min)
          param_maxs.append(param_max)
        elif row_param_index > col_param_index:
          self._plot_jpdf(ax, row_param_index, col_param_index)
          self._plot_kde(ax, row_param_index, col_param_index)
        else: ax.axis("off")
    for row_param_index in range(num_params):
      for col_param_index in range(num_params):
        ax = axs[row_param_index, col_param_index]
        if row_param_index == col_param_index:
          ax.set_xlim(param_mins[row_param_index], param_maxs[row_param_index])
        if row_param_index > col_param_index:
          ax.set_xlim(param_mins[col_param_index], param_maxs[col_param_index])
          ax.set_ylim(param_mins[row_param_index], param_maxs[row_param_index])
        if row_param_index == num_params-1: ax.set_xlabel(self.param_labels[col_param_index])
        if row_param_index < self.samples.shape[1]-1: ax.set_xticklabels([])
        if col_param_index == 0: ax.set_ylabel(self.param_labels[row_param_index])
        if col_param_index > 0: ax.set_yticklabels([])
    fig_name = f"{self.routine_name}_corner_plot.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)

  def _plot_jpdf(self, ax, row_param_index, col_param_index):
    row_data = self.samples[:, row_param_index]
    col_data = self.samples[:, col_param_index]
    bc_rows, bc_cols, estimated_jpdf = compute_stats.estimate_jpdf(data_x=col_data, data_y=row_data, num_bins=50)
    extent = [ bc_cols[0], bc_cols[-1], bc_rows[0], bc_rows[-1] ]
    ax.imshow(
      estimated_jpdf,
      extent = extent,
      origin = "lower",
      aspect = "auto",
      cmap   = "Blues"
    )

  def _plot_kde_per_slice(self, ax, row_param_index, col_param_index, num_points=100):
    data = self.samples[:, [col_param_index, row_param_index]]
    kde_2d = gaussian_kde(data.T)
    x = numpy.linspace(data[:, 0].min(), data[:, 0].max(), num_points)
    y = numpy.linspace(data[:, 1].min(), data[:, 1].max(), num_points)
    X, Y = numpy.meshgrid(x, y)
    positions = numpy.vstack([X.ravel(), Y.ravel()])
    Z = kde_2d(positions).reshape(X.shape)
    ax.contour(X, Y, Z, colors="red", linewidths=2.0)

  # def _plot_kde(self, ax, row_param_index, col_param_index, num_points=100):
  #   row_data = self.samples[:, row_param_index]
  #   col_data = self.samples[:, col_param_index]
  #   x = numpy.linspace(col_data.min(), col_data.max(), num_points)
  #   y = numpy.linspace(row_data.min(), row_data.max(), num_points)
  #   X, Y = numpy.meshgrid(x, y)
  #   grid_coords = numpy.column_stack([X.ravel(), Y.ravel()])
  #   param_means = self.samples.mean(axis=0)
  #   grid_full = numpy.tile(param_means, (grid_coords.shape[0], 1))
  #   grid_full[:, col_param_index] = grid_coords[:, 0]
  #   grid_full[:, row_param_index] = grid_coords[:, 1]
  #   Z = self.kde(grid_full.T).reshape(num_points, num_points)
  #   ax.contour(X, Y, Z, colors="black", linewidths=0.8)
  #   self._plot_kde_per_slice(ax, row_param_index, col_param_index, num_points=100)

  def _plot_kde(self, ax, row_param_index, col_param_index):
    print(f"Estimating KDE projection: axs[{row_param_index}][{col_param_index}]")
    Xi, Xj, Z = compute_2d_kde_projection(
        full_kde = self.kde,
        samples = self.samples,
        i=col_param_index,
        j=row_param_index,
        num_points=30,
        num_marginal_samples=50,
    )
    ax.contour(Xi, Xj, Z, colors="black", linewidths=1.0)
    self._plot_kde_per_slice(ax, row_param_index, col_param_index, num_points=100)

  def _compute_scaled_kde(self):
    print("Estimating KDE...")
    self.kde = gaussian_kde(self.samples.T, bw_method="scott")

def compute_2d_kde_projection(full_kde, samples, i, j, num_points, num_marginal_samples):
    ndim = samples.shape[1]
    other_indices = [k for k in range(ndim) if k != i and k != j]
    xi = numpy.linspace(samples[:, i].min(), samples[:, i].max(), num_points)
    xj = numpy.linspace(samples[:, j].min(), samples[:, j].max(), num_points)
    Xi, Xj = numpy.meshgrid(xi, xj)
    X_flat = Xi.ravel()
    Y_flat = Xj.ravel()
    n_grid = X_flat.shape[0]
    # Draw marginal samples
    marginal_values = samples[numpy.random.choice(samples.shape[0], size=num_marginal_samples)][..., other_indices]
    # Repeat grid and marginal samples to build full KDE input
    grid_points = numpy.zeros((n_grid * num_marginal_samples, ndim))
    grid_points[:, i] = numpy.repeat(X_flat, num_marginal_samples)
    grid_points[:, j] = numpy.repeat(Y_flat, num_marginal_samples)
    marginal_tiled = numpy.tile(marginal_values, (n_grid, 1))
    grid_points[:, other_indices] = marginal_tiled
    # Evaluate and reshape
    Z_vals = full_kde(grid_points.T)
    Z_avg = Z_vals.reshape(n_grid, num_marginal_samples).mean(axis=1).reshape(num_points, num_points)
    return Xi, Xj, Z_avg


## END OF MODULE