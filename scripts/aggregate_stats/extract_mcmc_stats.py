## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import manage_io, json_io
from jormi import ww_lists


def extract_from_mcmc_data(
    samples: numpy.ndarray,
    model: str,
) -> dict[str, numpy.ndarray]:
    init_energy = 10**samples[:, 0]
    sat_energy = 10**samples[:, 1]
    gamma_exp = samples[:, 2]
    nl_start_time = samples[:, 3]
    sat_start_time = samples[:, 4]
    if model == "linear": nl_exponent = numpy.ones_like(gamma_exp)
    elif model == "quadratic": nl_exponent = 2.0 * numpy.ones_like(gamma_exp)
    elif model == "free": nl_exponent = samples[:, 5]
    else: raise ValueError(f"Unknown model type: {model}")
    nl_start_energy = init_energy * numpy.exp(gamma_exp * nl_start_time)
    nl_duration = sat_start_time - nl_start_time
    gamma_nl = (sat_energy - nl_start_energy) / (nl_duration**nl_exponent)
    return {
        "gamma_exp": gamma_exp,
        "gamma_nl": gamma_nl,
        "nl_start_time": nl_start_time,
        "sat_start_time": sat_start_time,
        "nl_duration": nl_duration,
        "sat_energy": sat_energy,
        "nl_exponent": nl_exponent,
    }


class EnsembleAverager:
    model_types = [
        "free",
        "linear",
        "quadratic",
    ]
    binning_types = [
        "bin_per_t0",
        "100bins",
    ]
    quantity_keys = [
        "gamma_exp",
        "gamma_nl",
        "nl_duration",
        "sat_energy",
        "nl_exponent",
    ]

    def __init__(
        self,
        sim_directories: list,
    ) -> None:
        self.sim_directories = sim_directories
        self.fit_summary: dict[str, dict] = {}
        self.sim_params: dict[str, dict[str, float]] | None = None
        self.exracted_data = False

    def run(
        self,
    ) -> dict[str, dict | None]:
        ## for each fit-model
        for model_type in self.model_types:
            print("Processing model-fit:", model_type)
            print(" ")
            ## initialise quantities we want to accumulate over the different simulation instances
            combined_by_binning: dict[str, dict[str, list]] = {}
            ## loop over the different simulation instances
            for sim_directory in self.sim_directories:
                print("Looking at:", sim_directory)
                ## load sim meta data once
                sim_data_path = manage_io.combine_file_path_parts([sim_directory, "sim_data.json"])
                if not manage_io.does_file_exist(sim_data_path):
                    print(f"Missing sim_data.json for: {sim_directory}")
                    continue
                sim_data = json_io.read_json_file_into_dict(sim_data_path)
                target_Mach = sim_data["details"]["target_Mach"]
                target_Re = sim_data["details"]["target_Re"]
                for binning_type in self.binning_types:
                    mcmc_data_path = manage_io.combine_file_path_parts(
                        [
                            sim_directory,
                            model_type,
                            binning_type,
                            f"stage2_{model_type}_fitted_posterior_samples.npy",
                        ],
                    )
                    if not manage_io.does_file_exist(mcmc_data_path):
                        print(
                            f"Simulation does not have mcmc data fitted with `{model_type}` and `{binning_type}`\n",
                        )
                        continue
                    mcmc_data = numpy.load(mcmc_data_path)
                    extracted_data = extract_from_mcmc_data(mcmc_data, model_type)
                    if binning_type not in combined_by_binning:
                        combined_by_binning[binning_type] = {
                            quantity_key: []
                            for quantity_key in self.quantity_keys
                        }
                    for quantity_key in self.quantity_keys:
                        combined_by_binning[binning_type][quantity_key].append(
                            extracted_data[quantity_key],
                        )
                    ## only extract sim_params once (arbitrarily, from the linear fit)
                    if not self.exracted_data:
                        nu = 0.5 * target_Mach / target_Re
                        t_turb = sim_data["details"]["t_0"]
                        full_time_values = numpy.array(sim_data["time_series"]["time"])
                        full_Mach_energy = numpy.array(sim_data["time_series"]["Mach"])
                        median_nl_start_time = float(numpy.median(extracted_data["nl_start_time"]))
                        normalized_times: list[float] = [float(v) for v in full_time_values / t_turb]
                        start_index = ww_lists.get_index_of_first_crossing(values=normalized_times, target=5)
                        full_time_list: list[float] = [float(v) for v in full_time_values]
                        end_index = ww_lists.get_index_of_first_crossing(
                            values=full_time_list,
                            target=median_nl_start_time,
                        )
                        kinematic_Mach_values = full_Mach_energy[start_index:end_index]
                        kinematic_Re_values = 0.5 * kinematic_Mach_values / nu
                        self.sim_params = {
                            "Mach": {
                                "p16": float(numpy.percentile(kinematic_Mach_values, 16)),
                                "p50": float(numpy.percentile(kinematic_Mach_values, 50)),
                                "p84": float(numpy.percentile(kinematic_Mach_values, 84)),
                            },
                            "Re": {
                                "p16": float(numpy.percentile(kinematic_Re_values, 16)),
                                "p50": float(numpy.percentile(kinematic_Re_values, 50)),
                                "p84": float(numpy.percentile(kinematic_Re_values, 84)),
                            },
                        }
                        self.exracted_data = True
                    print(" ")
            self.fit_summary[model_type] = {}
            for binning_type, combined_data in combined_by_binning.items():
                self.fit_summary[model_type][binning_type] = {}
                for quantity_key, lists_of_samples in combined_data.items():
                    if not lists_of_samples:
                        self.fit_summary[model_type][binning_type][quantity_key] = {
                            "p16": None,
                            "p50": None,
                            "p84": None,
                        }
                        continue
                    flat_samples = numpy.concatenate(lists_of_samples)
                    self.fit_summary[model_type][binning_type][quantity_key] = {
                        "p16": float(numpy.percentile(flat_samples, 16)),
                        "p50": float(numpy.percentile(flat_samples, 50)),
                        "p84": float(numpy.percentile(flat_samples, 84)),
                    }
        return {
            "fit_summaries": self.fit_summary,
            "sim_params": self.sim_params,
        }


def main() -> None:
    script_dir = Path(__file__).parent
    datasets_dir = (script_dir / ".." / ".." / "datasets").resolve()
    output_summary_path = datasets_dir / "summary_stats.json"
    base_directory = datasets_dir / "sims"
    all_directories = manage_io.filter_directory(
        base_directory,
        req_include_words=["Mach", "Re", "Pm", "Nres"],
    )
    sim_suites = set([str(sim_directory).split("/")[-1].split("v")[0] for sim_directory in all_directories])
    all_results = {}
    for sim_suite in sorted(sim_suites):
        directories_in_suite = [
            sim_directory for sim_directory in all_directories if sim_suite in str(sim_directory)
        ]
        ## average over the different simulation instances (at a particular resolution)
        sim_averager = EnsembleAverager(directories_in_suite)
        all_results[sim_suite] = sim_averager.run()
    json_io.save_dict_to_json_file(
        file_path=output_summary_path,
        input_dict=all_results,
        overwrite=True,
    )


if __name__ == "__main__":
    main()

## } MODULE
