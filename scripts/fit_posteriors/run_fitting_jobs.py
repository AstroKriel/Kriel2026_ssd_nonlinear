## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

## personal
from jormi.ww_io import manage_io

##
## === GLOBAL PARAMS
##

ALLOW_OVERWRITE = False

##
## === CONSTANTS
##

SCRIPT_DIR = Path(__file__).parent
UV_PROJECT = (SCRIPT_DIR / ".." / "..").resolve()
SIMS_DIR = (SCRIPT_DIR / ".." / ".." / "datasets" / "sims").resolve()

class BinningConfig(TypedDict):
    tag: str
    num_bins: int | None


MODEL_TYPES = [
    "free",
    "linear",
    "quadratic",
]
BINNING_CONFIGS: list[BinningConfig] = [
    {"tag": "bin_per_t0", "num_bins": None},
    {"tag": "100bins",    "num_bins": 100},
]

##
## === HELPER FUNCTIONS
##


def output_exists(
    sim_directory: Path,
    model_name: str,
    binning_tag: str,
) -> bool:
    posterior_path = (
        sim_directory / model_name / binning_tag / f"stage2_{model_name}_fitted_posterior_samples.npy"
    )
    return posterior_path.exists()


def run_fit(
    sim_directory: Path,
    model_name: str,
    num_bins: int | None,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "fit_with_mcmc.py"),
        "-data_directory", str(sim_directory),
        "-model", model_name,
    ]
    if num_bins is not None:
        cmd += ["-num_bins", str(num_bins)]
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


##
## === MAIN PROGRAM
##


def main() -> None:
    all_sim_directories = manage_io.ItemFilter(
        req_include_words=["Mach", "Re", "Pm", "Nres"],
    ).filter(directory=SIMS_DIR)
    for sim_directory in sorted(all_sim_directories):
        for model_name in MODEL_TYPES:
            for binning_config in BINNING_CONFIGS:
                binning_tag: str = binning_config["tag"]
                num_bins: int | None = binning_config["num_bins"]
                if not ALLOW_OVERWRITE and output_exists(sim_directory, model_name, binning_tag):
                    print(f"Skipping (already fitted): {sim_directory.name} / {model_name} / {binning_tag}")
                    continue
                print(f"Fitting: {sim_directory.name} / {model_name} / {binning_tag}")
                run_fit(sim_directory, model_name, num_bins)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
