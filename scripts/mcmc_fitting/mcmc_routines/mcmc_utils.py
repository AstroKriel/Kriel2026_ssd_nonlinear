## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
import numpy

##
## === DATA HELPERS
##


def compute_binned_data(
    x_values: numpy.ndarray,
    y_values: numpy.ndarray,
    num_bins: int,
) -> dict[str, numpy.ndarray]:
    x_bin_edges = numpy.linspace(0, numpy.max(x_values), num_bins + 1)
    x_bin_centers = 0.5 * (x_bin_edges[1:] + x_bin_edges[:-1])
    x_bin_indices = numpy.digitize(x_values, x_bin_edges) - 1
    y_ave_s = numpy.zeros(num_bins)
    y_std_s = numpy.zeros(num_bins)
    log10_y_ave_s = numpy.zeros(num_bins)
    log10_y_std_s = numpy.zeros(num_bins)
    for bin_index in range(num_bins):
        bin_mask = (x_bin_indices == bin_index)
        if numpy.any(bin_mask):
            y_values_in_bin = numpy.array(y_values)[bin_mask]
            log10_y_values_in_bin = numpy.log10(y_values_in_bin)
            y_ave_s[bin_index] = numpy.mean(y_values_in_bin)
            y_std_s[bin_index] = numpy.std(y_values_in_bin)
            log10_y_ave_s[bin_index] = numpy.mean(log10_y_values_in_bin)
            log10_y_std_s[bin_index] = numpy.std(log10_y_values_in_bin)
        else:
            y_ave_s[bin_index] = numpy.nan
            y_std_s[bin_index] = numpy.nan
            log10_y_ave_s[bin_index] = numpy.nan
            log10_y_std_s[bin_index] = numpy.nan
    return {
        "x_bin_centers": x_bin_centers,
        "y_ave_s": y_ave_s,
        "y_std_s": y_std_s,
        "log10_y_ave_s": log10_y_ave_s,
        "log10_y_std_s": log10_y_std_s,
    }


def compute_median_params_from_kde(
    kde: Any,
    num_samples: int = 10000,
) -> tuple[float, ...]:
    samples = kde.resample(num_samples)
    return tuple(numpy.median(samples, axis=1))


##
## === PLOT HELPERS
##


def plot_param_percentiles(
    ax: Any,
    samples: numpy.ndarray,
    orientation: str,
) -> None:
    p16, p50, p84 = numpy.percentile(
        samples,
        [16, 50, 84],
    )
    if "h" in orientation.lower():
        ax_line = ax.axhline
        ax_span = ax.axhspan
    elif "v" in orientation.lower():
        ax_line = ax.axvline
        ax_span = ax.axvspan
    else:
        raise ValueError("`orientation` must either be `horizontal` (`h`) or `vertical` (`v`).")
    ax_line(
        p50,
        color="green",
        ls=":",
        lw=1.5,
        zorder=5,
    )
    ax_span(
        p16,
        p84,
        color="green",
        ls="-",
        lw=1.5,
        alpha=0.3,
        zorder=4,
    )


## } MODULE
