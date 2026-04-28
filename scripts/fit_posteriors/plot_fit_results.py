## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse
from pathlib import Path
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

## personal
from jormi.ww_io import json_io

## local
from mcmc_routines.mcmc_stage_1 import Stage1MCMCRoutine
from mcmc_routines.mcmc_stage_2 import (
    Stage2MCMCRoutine_free,
    Stage2MCMCRoutine_linear,
    Stage2MCMCRoutine_quadratic,
)
from mcmc_routines.mcmc_base import BaseMCMCRoutine
from mcmc_routines.mcmc_utils import compute_binned_data
from mcmc_routines.plot_final_fits import PlotFinalFits

##
## === CONSTANTS
##

STAGE2_CLASSES = {
    "free": Stage2MCMCRoutine_free,
    "linear": Stage2MCMCRoutine_linear,
    "quadratic": Stage2MCMCRoutine_quadratic,
}
MAX_KDE_SAMPLES = 5_000

##
## === HELPERS
##


def build_binned_data(
    *,
    data_dict: dict[str, Any],
    binning_tag: str,
) -> dict[str, Any]:
    """Bin the time-series from `data_dict` according to `binning_tag`."""
    full_time_values = numpy.array(data_dict["time_series"]["time"])
    full_magnetic_energy = numpy.array(data_dict["time_series"]["Emag"])
    t_turb = data_dict["details"]["t_0"]
    if binning_tag == "bin_per_t0":
        num_bins = int(numpy.max(full_time_values) / t_turb)
    else:
        num_bins = int(binning_tag.removesuffix("bins"))
    return compute_binned_data(
        x_values=full_time_values,
        y_values=full_magnetic_energy,
        num_bins=num_bins,
    )


def load_posterior_data(
    *,
    routine: BaseMCMCRoutine,
    fit_dir: Path,
    routine_name: str,
) -> None:
    """Populate `routine` with saved MCMC samples and recomputed KDEs from disk."""
    rng = numpy.random.default_rng(seed=0)

    def _subsample(
        samples: NDArray[Any],
    ) -> NDArray[Any]:
        if len(samples) <= MAX_KDE_SAMPLES:
            return samples
        return samples[rng.choice(
            len(samples),
            size=MAX_KDE_SAMPLES,
            replace=False,
        )]

    fitted_samples: NDArray[Any] = numpy.load(
        fit_dir / f"{routine_name}_fitted_posterior_samples.npy",
    )
    routine.fitted_posterior_samples = fitted_samples
    routine.num_params = int(fitted_samples.shape[1])
    routine.fitted_log_likelihoods = numpy.load(
        fit_dir / f"{routine_name}_fitted_log_likelihoods.npy",
    )
    output_path = fit_dir / f"{routine_name}_output_posterior_samples.npy"
    output_samples: NDArray[Any] = (numpy.load(output_path) if output_path.is_file() else fitted_samples)
    routine.output_posterior_samples = output_samples
    chain_path = fit_dir / f"{routine_name}_raw_chain.npy"
    if chain_path.is_file():
        raw_chain: NDArray[Any] = numpy.load(chain_path)
        routine.raw_chain = raw_chain
        routine.num_walkers = int(raw_chain.shape[1])
        routine.num_steps = int(raw_chain.shape[0])
    routine.fitted_posterior_kde = gaussian_kde(
        _subsample(fitted_samples).T,
        bw_method="scott",
    )
    routine.output_posterior_kde = gaussian_kde(
        _subsample(output_samples).T,
        bw_method="scott",
    )
    _, routine.output_param_labels = routine.get_output_params()


##
## === PROGRAM MAIN
##


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-render diagnostic plots from a saved fit result without re-running the MCMC.",
    )
    parser.add_argument(
        "--fit_dir",
        required=True,
        help="Path to the fit output directory (e.g. datasets/sims/.../linear/bin_per_t0).",
    )
    args = parser.parse_args()
    fit_dir = Path(args.fit_dir).resolve()
    if not fit_dir.is_dir():
        raise FileNotFoundError(f"Fit directory not found: {fit_dir}")
    binning_tag = fit_dir.name
    model_name = fit_dir.parent.name
    data_directory = fit_dir.parent.parent
    if model_name not in STAGE2_CLASSES:
        raise ValueError(
            f"Unrecognised model `{model_name}` in path. Expected one of: {list(STAGE2_CLASSES)}.",
        )
    data_dict = json_io.read_json_file_into_dict(data_directory / "sim_data.json")
    binned_data = build_binned_data(
        data_dict=data_dict,
        binning_tag=binning_tag,
    )
    ## build stage 1 shell and plot saved results
    stage1_mcmc = Stage1MCMCRoutine(
        output_directory=fit_dir,
        time_values=binned_data["x_bin_centers"],
        ave_log10_energy_values=binned_data["log10_y_ave_s"],
        std_log10_energy_values=binned_data["log10_y_std_s"],
        plot_posterior_kde=True,
    )
    load_posterior_data(
        routine=stage1_mcmc,
        fit_dir=fit_dir,
        routine_name="stage1",
    )
    stage1_mcmc.make_plots()
    ## build stage 2 shell and plot saved results
    stage2_mcmc = STAGE2_CLASSES[model_name](
        output_directory=fit_dir,
        time_values=binned_data["x_bin_centers"],
        ave_energy_values=binned_data["y_ave_s"],
        std_energy_values=binned_data["y_std_s"],
        plot_posterior_kde=True,
    )
    load_posterior_data(
        routine=stage2_mcmc,
        fit_dir=fit_dir,
        routine_name=f"stage2_{model_name}",
    )
    stage2_mcmc.make_plots()
    PlotFinalFits(stage2_mcmc).plot()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
