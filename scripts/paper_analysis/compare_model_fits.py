import csv
import math
import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files

K = 5  # both models have 5 params
MODEL_ORDER = ["linear", "quadratic"]


def safe_len(x):
    try:
        return len(x)
    except Exception:
        return None


def akaike_weights(aic_values: dict[str, float]) -> dict[str, float]:
    vals = list(aic_values.values())
    if not vals: return {}
    min_aic = min(vals)
    w_unnorm = {m: math.exp(-0.5 * (aic - min_aic)) for m, aic in aic_values.items()}
    denom = sum(w_unnorm.values())
    return {m: (w / denom if denom > 0 else 0.0) for m, w in w_unnorm.items()}


def select_best(weights: dict[str, float]) -> str | None:
    return max(weights.items(), key=lambda kv: kv[1])[0] if weights else None


def expected_best_model(target_Mach: float) -> str:
    return "quadratic" if (target_Mach > 1.0) else "linear"


def compute_ics(max_ll: float, k: int, N: int | None):
    aic = 2 * k - 2.0 * max_ll
    out = {"AIC": aic, "AICc": float("nan"), "BIC": float("nan")}
    if isinstance(N, int) and N > 0:
        if N > (k + 1):
            out["AICc"] = aic + (2.0 * k * (k + 1)) / (N - k - 1)
        out["BIC"] = k * math.log(N) - 2.0 * max_ll
    return out


def load_max_ll(sim_directory: Path, model_name: str) -> float | None:
    mcmc_results_path = io_manager.combine_file_path_parts(
        [
            sim_directory,
            model_name,
            "bin_per_t0",
            f"stage2_{model_name}_fitted_log_likelihoods.npy",
        ],
    )
    if not io_manager.does_file_exist(mcmc_results_path):
        return None
    arr = numpy.load(mcmc_results_path)
    return float(numpy.max(arr)) if arr.size else None


def get_fit_sample_size(sim_data: dict) -> int | None:
    try:
        N = safe_len(sim_data["measured_data"]["time_values"])
        return int(N) if N is not None else None
    except Exception:
        return None


def main():
    base_dir = Path(
        "/Users/necoturb/Documents/Codes/asgard/mimir/kriel_2025_ssd_nl/datasets_v2/",
    ).resolve()
    out_csv = base_dir / "model_comparison_summary.csv"

    sim_directories = io_manager.ItemFilter(
        include_string=["Mach", "Re", "Pm", "Nres576"],
    ).filter(directory=base_dir)

    rows = []
    total = 0
    ok_expected_aic = 0
    ok_expected_aicc = 0
    ok_expected_bic = 0

    for sim_directory in sim_directories:
        sim_data_path = sim_directory / "dataset.json"
        if not io_manager.does_file_exist(sim_data_path):
            continue

        sim_data = json_files.read_json_file_into_dict(sim_data_path, verbose=False)
        target_Mach = float(sim_data["plasma_params"]["target_Mach"])
        target_Re = float(sim_data["plasma_params"]["target_Re"])
        if target_Mach < 1: continue
        if target_Re < 1000: continue

        N = get_fit_sample_size(sim_data)

        ic_AIC, ic_AICc, ic_BIC = {}, {}, {}
        maxlls = {}
        for model_name in MODEL_ORDER:
            max_ll = load_max_ll(sim_directory, model_name)
            if max_ll is None:
                continue
            maxlls[model_name] = max_ll
            ics = compute_ics(max_ll=max_ll, k=K, N=N)
            ic_AIC[model_name] = ics["AIC"]
            ic_AICc[model_name] = ics["AICc"]
            ic_BIC[model_name] = ics["BIC"]

        if len(ic_AIC) < 2:
            print("Missing results for:", sim_directory)
            continue

        wAIC = akaike_weights(ic_AIC)
        wAICc = akaike_weights({m: v for m, v in ic_AICc.items() if not math.isnan(v)})
        if len(ic_BIC) >= 2 and all(not math.isnan(v) for v in ic_BIC.values()):
            min_bic = min(ic_BIC.values())
            wBIC = {m: math.exp(-0.5 * (ic_BIC[m] - min_bic)) for m in ic_BIC}
            denom = sum(wBIC.values())
            wBIC = {m: (w / denom if denom > 0 else 0.0) for m, w in wBIC.items()}
            best_BIC = min(ic_BIC.items(), key=lambda kv: kv[1])[0]
        else:
            wBIC, best_BIC = {}, None

        best_AIC = select_best(wAIC)
        best_AICc = select_best(wAICc) if wAICc else None

        expected = expected_best_model(target_Mach)

        total += 1
        ok_expected_aic += int(best_AIC == expected)
        ok_expected_aicc += int(best_AICc == expected) if best_AICc is not None else 0
        ok_expected_bic += int(best_BIC == expected) if best_BIC is not None else 0

        row = {
            "sim_path": str(sim_directory),
            "Mach": target_Mach,
            "Re": target_Re,
            "N": (N if N is not None else -1),
            "maxLL_linear": maxlls.get("linear", float("nan")),
            "maxLL_quadratic": maxlls.get("quadratic", float("nan")),
            "AIC_linear": ic_AIC.get("linear", float("nan")),
            "AIC_quadratic": ic_AIC.get("quadratic", float("nan")),
            "wAIC_linear": wAIC.get("linear", float("nan")),
            "wAIC_quadratic": wAIC.get("quadratic", float("nan")),
            "best_AIC": best_AIC or "",
            "AICc_linear": ic_AICc.get("linear", float("nan")),
            "AICc_quadratic": ic_AICc.get("quadratic", float("nan")),
            "wAICc_linear": (wAICc.get("linear", float("nan")) if wAICc else float("nan")),
            "wAICc_quadratic": (wAICc.get("quadratic", float("nan")) if wAICc else float("nan")),
            "best_AICc": best_AICc or "",
            "BIC_linear": ic_BIC.get("linear", float("nan")),
            "BIC_quadratic": ic_BIC.get("quadratic", float("nan")),
            "best_BIC": best_BIC or "",
            "expected_by_Mach": expected,
            "match_AIC": int(best_AIC == expected) if best_AIC else -1,
            "match_AICc": int(best_AICc == expected) if best_AICc else -1,
            "match_BIC": int(best_BIC == expected) if best_BIC else -1,
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["Mach"], r["Re"]))

    fieldnames = [
        "sim_path",
        "Mach",
        "Re",
        "N",
        "maxLL_linear",
        "maxLL_quadratic",
        "AIC_linear",
        "AIC_quadratic",
        "wAIC_linear",
        "wAIC_quadratic",
        "best_AIC",
        "AICc_linear",
        "AICc_quadratic",
        "wAICc_linear",
        "wAICc_quadratic",
        "best_AICc",
        "BIC_linear",
        "BIC_quadratic",
        "best_BIC",
        "expected_by_Mach",
        "match_AIC",
        "match_AICc",
        "match_BIC",
    ]
    out_csv = base_dir / "model_comparison_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    def pct(x, n):
        return f"{(100.0 * x / n):.1f}%" if n > 0 else "n/a"

    print(f"\nWrote: {out_csv}")
    print(f"Comparisons: {total}")
    print(f"Matches (AIC):  {ok_expected_aic}/{total}  = {pct(ok_expected_aic,total)}")
    print(f"Matches (AICc): {ok_expected_aicc}/{total} = {pct(ok_expected_aicc,total)}")
    print(f"Matches (BIC):  {ok_expected_bic}/{total}  = {pct(ok_expected_bic,total)}")


if __name__ == "__main__":
    main()
