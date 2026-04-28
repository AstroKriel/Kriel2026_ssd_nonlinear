## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import manage_io, json_io


def get_max_loglikelihood(
    sim_dir: Path,
    model_name: str,
) -> float | None:
    ll_path = sim_dir / model_name / "bin_per_t0" / f"stage2_{model_name}_fitted_log_likelihoods.npy"
    if not ll_path.exists():
        return None
    ll_data = numpy.load(ll_path)
    return numpy.max(ll_data)


def get_linear_model_weight(
    max_ll_linear: float | None,
    max_ll_quadratic: float | None,
) -> float:
    if max_ll_linear is None or max_ll_quadratic is None:
        return 0.5
    aic_linear = -2 * max_ll_linear
    aic_quadratic = -2 * max_ll_quadratic
    min_aic = numpy.min([aic_linear, aic_quadratic])
    return numpy.exp(
        -0.5 * (aic_linear - min_aic),
    ) / (numpy.exp(-0.5 * (aic_linear - min_aic)) + numpy.exp(-0.5 * (aic_quadratic - min_aic)))


def main() -> None:
    script_dir = Path(__file__).parent
    dataset_dir = (script_dir / ".." / ".." / "datasets").resolve()
    sim_dirs = manage_io.filter_directory(
        dataset_dir / "sims",
        req_include_words="Mach",
        include_files=False,
    )
    num_sims = 0
    agreement = 0
    for sim_dir in sim_dirs:
        sim_data = json_io.read_json_file_into_dict(sim_dir / "sim_data.json", verbose=False)
        target_Mach = float(sim_data["details"]["target_Mach"])
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

## } SCRIPT
