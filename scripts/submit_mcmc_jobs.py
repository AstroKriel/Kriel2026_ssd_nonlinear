# ###############################################################
# DEPENDANCIES
# ###############################################################

import sys
from pathlib import Path

from jormi.ww_io import io_manager, json_files, shell_manager
from jormi.ww_jobs import pbs_job_manager


# ###############################################################
# HELPER FUNCTIONS
# ###############################################################
def submit_job(data_directory, queued_job_tags) -> None:
    data_path = io_manager.combine_file_path_parts([data_directory, "dataset.json"])
    data_dict = json_files.read_json_file_into_dict(data_path, verbose=False)
    command_path = io_manager.combine_file_path_parts(["fit_with_mcmc.py"])
    sim_name = data_dict["sim_name"]
    job_tag = f"{sim_name}_mcmc"
    job_path = pbs_job_manager.create_pbs_job_script(
        system_name="gadi",
        directory=data_directory,
        file_name="mcmc_fit.sh",
        command=f"uv run {command_path} -data_directory {data_directory}",
        tag_name=job_tag,
        queue_name="normal",
        compute_group_name="jh2",
        num_procs=8,
        wall_time_hours=2,
        storage_group_name="jh2",
        email_address="neco.kriel@anu.edu.au",
        email_on_start=False,
        email_on_finish=False,
        verbose=True,
    )
    if job_tag in queued_job_tags:
        print(f"Job ({job_tag}) is already in the pbs queue.")
        return
    shell_manager.execute_shell_command(
        command=f"qsub {job_path}",
        working_directory=data_directory,
        timeout_seconds=30,
    )


# ###############################################################
# MAIN PROGRAM
# ###############################################################


def main() -> None:
    # collect data directories
    base_directory = Path("../data/").resolve()
    data_directories = io_manager.ItemFilter(
        include_string=["Mach", "Re", "Pm", "Nres"]
    ).filter(directory=base_directory)
    # collect queued jobs
    queued_jobs = pbs_job_manager.get_list_of_queued_jobs()
    queued_job_tags = [job_tag for _, job_tag in queued_jobs]
    # ## delete mcmc jobs from queue
    # [
    #   shell_manager.execute_shell_command(command=f"qdel {job_id}", timeout_seconds=30)
    #   for job_id, job_tag in queued_jobs
    #   if "_mcmc" in job_tag
    # ]
    # create and submit mcmc job
    for data_directory in data_directories:
        submit_job(data_directory, queued_job_tags)
        print(" ")


# ###############################################################
# SCRIPT ENTRY POINT
# ###############################################################
if __name__ == "__main__":
    main()
    sys.exit(0)


# END OF SCRIPT

