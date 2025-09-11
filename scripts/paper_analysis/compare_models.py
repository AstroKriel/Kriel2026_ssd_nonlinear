import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files


def get_max_loglikelihood(
    sim_dir: Path,
    model_name: str,
) -> float | None:
    ll_path = io_manager.combine_file_path_parts(
        [sim_dir, model_name, "bin_per_t0", f"stage2_{model_name}_fitted_log_likelihoods.npy"],
    )
    ll_data = numpy.load(ll_path)
    return numpy.max(ll_data)


def get_linear_model_weight(
    max_ll_linear: float,
    max_ll_quadratic: float,
) -> float:
    aic_linear = -2 * max_ll_linear
    aic_quadratic = -2 * max_ll_quadratic
    min_aic = numpy.min([aic_linear, aic_quadratic])
    return numpy.exp(-0.5 * (aic_linear - min_aic)) / (
        numpy.exp(-0.5 * (aic_linear - min_aic)) + numpy.exp(-0.5 * (aic_quadratic - min_aic))
    )


def main():
    script_dir = Path(__file__).parent
    dataset_dir = (script_dir / ".." / "datasets").resolve()
    sim_dirs = io_manager.ItemFilter(
        include_string="Mach",
        include_files=False,
        include_folders=True,
    ).filter(
        directory=dataset_dir / "backup"
    )
    num_sims = 0
    agreement = 0
    for sim_dir in sim_dirs:
        sim_data = json_files.read_json_file_into_dict(sim_dir / "dataset.json", verbose=False)
        target_Mach = float(sim_data["plasma_params"]["target_Mach"])
        max_ll_linear = get_max_loglikelihood(sim_dir, "linear")
        max_ll_quadratic = get_max_loglikelihood(sim_dir, "quadratic")
        linear_model_weight = get_linear_model_weight(max_ll_linear, max_ll_quadratic)
        best_model = "linear" if linear_model_weight >= 0.5 else "quadratic"
        expected_model = "quadratic" if target_Mach > 1.0 else "linear"
        model_agreement = int(best_model == expected_model)
        num_sims += 1
        agreement += model_agreement
    print(f"Compared: {num_sims}")
    print(f"Agreement: {agreement}/{num_sims} = {100 * agreement / num_sims:.1f}%")


if __name__ == "__main__":
    main()
