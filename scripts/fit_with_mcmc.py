# ###############################################################
# DEPENDENCIES
# ###############################################################

from typing import TYPE_CHECKING
from pathlib import Path
import argparse

import numpy as np
from jormi.utils import list_utils
from jormi.ww_io import io_manager, json_files

from scripts import my_mcmc_routine

if TYPE_CHECKING:
    from scipy.stats import gaussian_kde

# ###############################################################
# HELPER FUNCTIONS
# ###############################################################


def compute_median_params_from_kde(
    kde: "gaussian_kde", num_samples: int = 10000
) -> "tuple[float, ...]":
    samples = kde.resample(num_samples)
    return tuple(np.median(samples, axis=1))


# ###############################################################
# PROGRAM MAIN
# ###############################################################


def main() -> None:
    # collect user arguments
    parser = argparse.ArgumentParser(description="Run MCMC fitting routine.")
    # SUGGESTION: If you just have the arg name without `-` it is a positional argument and is required by default
    parser.add_argument(
        "data_directory", type=str
    )  # TODO(AstroKriel): Add help documentation
    data_directory = Path(parser.parse_args().data_directory).resolve()
    # read in magnetic energy evolution
    data_path = io_manager.combine_file_path_parts([
        data_directory,
        "dataset.json",
    ])  # TODO(AstroKriel): Change combine_file_path_parts's type def from `list[str] | list[Path]` to `list[str | Path]`
    data_dict = json_files.read_json_file_into_dict(data_path)
    # TODO(AstroKriel): Not worth it now, but in the future, check out [Pydantic](https://docs.pydantic.dev/dev/). Let's you check early that the user has provided the right keys and values in the json file.
    #                   For instance, try running your code on `test/dataset.json`. The error you get isn't very helpful to a user.
    x_values = data_dict["interp_data"]["time"]
    y_values = data_dict["interp_data"]["magnetic_energy"]

    # --- Stage 1 ---
    # TODO(AstroKriel): Move into it's own function. Allows you to run stage 1 independently of stage 2. Useful for testing and makes your code more modular.

    # build initial guess for stage 1: exponential + saturation
    stage1_initial_params = (
        -20,  # log10(E_init)
        0.5,  # log10(E_sat)
        0.5 * np.max(x_values),  # gammma
    )  # TODO(AstroKriel): Make these options in the json file

    # run stage 1 fitter
    stage1_mcmc = my_mcmc_routine.MCMCStage1Routine(
        output_directory=data_directory,
        x_values=x_values,
        y_values=y_values,
        initial_params=stage1_initial_params,
        plot_posterior_kde=True,
    )
    stage1_mcmc.estimate_posterior()

    # extract key outputs from stage 1
    stage1_median_transition_time = np.median(
        stage1_mcmc.fitted_posterior_samples[:, 2]
    )

    # --- Stage 2 ---
    # TODO(AstroKriel): Move into it's own function. Allows you to run stage 2 independently of stage 1. Useful for testing and makes your code more modular.

    # build initial guess for stage 2: exponential + linear backreaction + saturation
    stage2_prior_kde = stage1_mcmc.output_posterior_kde
    stage1_median_output_params = compute_median_params_from_kde(stage2_prior_kde)

    stage2_initial_params = (
        stage1_median_output_params[0],  # log10(E_init)
        stage1_median_output_params[1],  # log10(E_sat)
        stage1_median_output_params[2],  # gammma
        0.5 * stage1_median_transition_time,  # t_nl
        0.5 * (np.max(x_values) + stage1_median_transition_time),  # t_sat
    )  # TODO(AstroKriel): Make these options in the json file

    approx_transition_index = list_utils.get_index_of_closest_value(
        x_values, stage1_median_transition_time
    )  # TODO(AstroKriel): fix get_index_of_closest_value type signature
    # TODO(AstroKriel): More comments to describe what these do. Lots of comments in main avoids users having to dig further into your code.

    stage2_likelihood_sigma = np.std(
        y_values[approx_transition_index:]
    )  # TODO(AstroKriel): More comments to describe what these do. Lots of comments in main avoids users having to dig further into your code.

    # run stage 2 fitter
    stage2_mcmc = my_mcmc_routine.MCMCStage2Routine(
        output_directory=data_directory,
        x_values=x_values,
        y_values=y_values,
        initial_params=stage2_initial_params,
        prior_kde=stage2_prior_kde,
        likelihood_sigma=stage2_likelihood_sigma,
        plot_posterior_kde=True,
    )
    stage2_mcmc.estimate_posterior()

    # --- Analysis ---
    # TODO(AstroKriel): Move into it's own function. Allows you to run analysis independently of stage 1 and 2. Useful for testing and makes your code more modular.

    # plot the measured vs modelled energy evolution (both in linear and log10-transformed domains)
    my_mcmc_routine.plot_final_fits.PlotFinalFits(stage2_mcmc).plot()


# ###############################################################
# SCRIPT ENTRY POINT
# ###############################################################

if __name__ == "__main__":
    main()


# END OF SCRIPT
