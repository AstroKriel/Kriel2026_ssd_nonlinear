## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any, final

## third-party
import numpy

## personal
from jormi.ww_plots import manage_plots
from jormi.ww_io import manage_io

##
## === PLOTTING ROUTINE
##


@final
class PlotFinalFits:

    def __init__(
        self,
        mcmc_routine: Any,
        num_curves: int = 100,
    ) -> None:
        self.num_curves = num_curves
        self.output_directory = mcmc_routine.output_directory
        self.x_values = mcmc_routine.x_values
        self.y_values = mcmc_routine.y_values
        self.fitted_posterior_samples = mcmc_routine.fitted_posterior_samples
        self._mcmc_routine = mcmc_routine

    def plot(
        self,
    ) -> None:
        fig, axs = manage_plots.create_figure(num_rows=2, num_cols=1, share_x=True)
        axs = axs[:, 0]
        self._plot_data(axs)
        self._plot_model(axs)
        self._label_plot(axs)
        fig_name = f"final_fit.png"
        fig_file_path = manage_io.combine_file_path_parts([self.output_directory, fig_name])
        manage_plots.save_figure(fig, fig_file_path, verbose=True)

    def _plot_data(
        self,
        axs: Any,
    ) -> None:
        style = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
        axs[0].plot(self.x_values, numpy.log10(self.y_values), **style)
        axs[1].plot(self.x_values, self.y_values, **style)
        axs[1].axhline(y=0, color="black", ls="--", lw=1.5, zorder=0)

    def _plot_model(
        self,
        axs: Any,
    ) -> None:
        rng = numpy.random.default_rng(seed=42)
        num_samples = self.fitted_posterior_samples.shape[0]
        random_indices = rng.choice(
            num_samples,
            size=min(self.num_curves, num_samples),
            replace=False,
        )
        modelled_curves = numpy.array(
            [
                self._mcmc_routine._model(self.fitted_posterior_samples[sample_index]).squeeze()
                for sample_index in random_indices
            ],
        )
        p16, p84 = numpy.percentile(modelled_curves, [16, 84], axis=0)
        axs[0].fill_between(
            self.x_values,
            numpy.log10(p16),
            numpy.log10(p84),
            color="red",
            alpha=0.25,
            zorder=3,
        )
        axs[1].fill_between(self.x_values, p16, p84, color="red", alpha=0.25, zorder=3)

    def _label_plot(
        self,
        axs: Any,
    ) -> None:
        axs[0].set_ylabel(r"$\log_{10}(E_{\mathrm{mag}})$")
        axs[1].set_ylabel(r"$E_{\mathrm{mag}}$")
        axs[1].set_xlabel(r"time")


## } MODULE
