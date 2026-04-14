## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import TypedDict

## personal
from jormi.ww_io import manage_io
from jormi.ww_jobs import pbs_manager

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
        sim_directory
        / model_name
        / binning_tag
        / f"stage2_{model_name}_fitted_posterior_samples.npy"
    )
    return posterior_path.exists()


def submit_job(
    data_directory: Path,
    model_name: str,
    queued_job_tags: list[str],
    num_bins: int | None,
) -> None:
    binning_tag = "bin_per_t0" if (num_bins is None) else f"{num_bins}bins"
    if not ALLOW_OVERWRITE and output_exists(data_directory, model_name, binning_tag):
        print(
            f"Skipping (already fitted): {data_directory.name} / {model_name} / {binning_tag}"
        )
        return
    tag_name = f"{data_directory.name}_{model_name}_{binning_tag}"
    if tag_name in queued_job_tags:
        print(f"Skipping (already in queue): {tag_name}")
        return
    fitting_cmd_parts = [
        f"uv run --project {UV_PROJECT} python fit_with_mcmc.py",
        f"-data_directory {data_directory}",
        f"-model {model_name}",
    ]
    if num_bins is not None:
        fitting_cmd_parts.append(f"-num_bins {num_bins}")
    main_command = f"cd {SCRIPT_DIR} && " + " ".join(fitting_cmd_parts)
    file_name = f"{tag_name}.sh"
    pbs_manager.create_pbs_job_script(
        system_name="gadi",
        directory=data_directory,
        file_name=file_name,
        main_command=main_command,
        tag_name=tag_name,
        queue_name="normal",
        num_procs=1,
        wall_time_hours=4,
    )
    pbs_manager.submit_job(
        directory=data_directory,
        file_name=file_name,
    )
    queued_job_tags.append(tag_name)


##
## === MAIN PROGRAM
##


def main() -> None:
    all_sim_directories = manage_io.ItemFilter(
        req_include_words=["Mach", "Re", "Pm", "Nres"],
    ).filter(directory=SIMS_DIR)
    current_queue = pbs_manager.get_list_of_queued_jobs()
    queued_job_tags: list[str] = (
        [job_tag for _, job_tag in current_queue] if current_queue else []
    )
    for sim_directory in sorted(all_sim_directories):
        for model_name in MODEL_TYPES:
            for binning_config in BINNING_CONFIGS:
                submit_job(
                    data_directory=sim_directory,
                    model_name=model_name,
                    queued_job_tags=queued_job_tags,
                    num_bins=binning_config["num_bins"],
                )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
