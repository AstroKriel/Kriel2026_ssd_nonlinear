## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import sys
from pathlib import Path

## personal
from jormi.ww_io import manage_io, json_io, manage_shell
from jormi.ww_jobs import pbs_manager

SCRIPT_DIR = Path(__file__).parent
UV_PROJECT = SCRIPT_DIR.parent

##
## === HELPER FUNCTIONS
##


def submit_job(
    data_directory: Path,
    model_name: str,
    queued_job_tags: list[str],
    num_bins: int | None = None,
) -> None:
    data_path = manage_io.combine_file_path_parts([data_directory, "sim_data.json"])
    data_dict = json_io.read_json_file_into_dict(data_path, verbose=False)
    sim_name = data_dict["details"]["name"]
    bin_tag = f"_{num_bins}bins" if (num_bins is not None) else ""
    job_tag = f"{sim_name}_mcmc_{model_name}{bin_tag}"
    if job_tag in queued_job_tags:
        print(f"Job ({job_tag}) is already in the pbs queue.")
        return
    job_directory = manage_io.combine_file_path_parts([SCRIPT_DIR.parent.parent / "mcmc_jobs", sim_name])
    manage_io.init_directory(job_directory)
    command_path = (SCRIPT_DIR / "fit_with_mcmc.py").resolve()
    manage_io.does_file_exist(
        file_path=command_path,
        raise_error=True,
    )
    bin_flag = f" -num_bins {num_bins}" if (num_bins is not None) else ""
    command = (
        f'uv run --project "{UV_PROJECT}" '
        f'python {command_path} -data_directory {data_directory} -model {model_name}{bin_flag}'
    )
    job_path = pbs_manager.create_pbs_job_script(
      system_name        = "gadi",
      directory          = job_directory,
      file_name          = f"mcmc_fit_{model_name}{bin_tag}.sh",
      main_command       = command,
      tag_name           = job_tag,
      queue_name         = "normal", # "rsaa",
      compute_group_name = "jh2", # "mk27",
      num_procs          = 2,
      wall_time_hours    = 24,
      storage_group_name = "jh2",
      email_address      = "neco.kriel@anu.edu.au",
      email_on_start     = False,
      email_on_finish    = False,
      verbose            = True,
    )
    manage_shell.execute_shell_command(
        command=f"qsub {job_path}",
        working_directory=job_directory,
        timeout_seconds=30,
    )


##
## === MAIN PROGRAM
##


def main() -> None:
    ## collect data directories
    base_directory = (SCRIPT_DIR.parent.parent / "datasets" / "sims").resolve()
    data_directories = manage_io.ItemFilter(
        req_include_words=["Mach", "Re", "Pm", "Nres"],
    ).filter(
        directory=base_directory,
    )
    [print(data_directory) for data_directory in data_directories]
    print(" ")
    ## collect queued jobs
    queued_jobs = pbs_manager.get_list_of_queued_jobs() or []
    queued_job_tags = [job_tag for _, job_tag in queued_jobs]
    ## delete mcmc jobs from queue
    # [
    #   manage_shell.execute_shell_command(command=f"qdel {job_id}", timeout_seconds=30)
    #   for job_id, job_tag in queued_jobs
    #   if "_mcmc_" in job_tag
    # ]
    ## create and submit mcmc job
    for data_directory in data_directories:
        for model_name in [
                "linear",
                "quadratic",
                "free",
        ]:
            for num_bins in [100, None]:
                submit_job(data_directory, model_name, queued_job_tags, num_bins)
                print(" ")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()
    sys.exit(0)

## } SCRIPT
