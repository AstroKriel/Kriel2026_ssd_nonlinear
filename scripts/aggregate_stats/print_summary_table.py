## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
import math

## personal
from jormi.ww_io import json_io

##
## === HELPER FUNCTIONS
##


def sci_basis(
    v: float,
) -> tuple[float, int]:
    """Return (mantissa, exponent) so that v = mantissa * 10**exponent and 1 <= |mantissa| < 10."""
    if not math.isfinite(v) or v == 0:
        return 0.0, 0
    k = int(
        math.floor(
            math.log10(
                abs(
                    v,
                ),
            ),
        ),
    )
    m = v / (10**k)
    if abs(m) < 1:
        k -= 1
        m = v / (10**k)
    return m, k


def decimals_from_max_err(
    e: float,
) -> int:
    """
  One significant digit for the largest uncertainty.
  decimals = max(0, -floor(log10(e))).
  """
    if not math.isfinite(e) or e <= 0:
        return 0
    k = math.floor(
        math.log10(
            e,
        ),
    )
    return max(0, -int(k))


def round_fixed(
    x: float,
    decimals: int,
) -> str:
    """Format with a fixed number of decimals (no scientific notation)."""
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(x)


def fmt_value_sci(
    v: float,
    sig: int = 2,
) -> str:
    """Format a plain value (no errors) in scientific notation for LaTeX math."""
    if v == 0 or not math.isfinite(v):
        return "$0$"
    k = int(
        math.floor(
            math.log10(
                abs(
                    v,
                ),
            ),
        ),
    )
    m = v / (10**k)
    m_str = f"{m:.{sig}g}"
    if k == 0:
        return f"${m_str}$"
    return f"${m_str}\\times 10^{{{k}}}$"


def from_log10(
    p50: float,
    lo: float,
    hi: float,
) -> tuple[float, float, float]:
    """
  Convert log10 median and asymmetric deltas to linear domain:
    v      = 10**p50
    err_lo = v - 10**(p50 - lo)
    err_hi = 10**(p50 + hi) - v
  """
    v = 10.0**p50
    vlo = 10.0**(p50 - lo)
    vhi = 10.0**(p50 + hi)
    return v, (v - vlo), (vhi - v)


def clip_uncertainty(
    val_str: str,
    decimals: int,
) -> str:
    """
  Ensure uncertainty string is never '0.0' etc.
  Clip to at least 0.1 (with given decimals).
  """
    if float(val_str) == 0.0:
        return f"{0.1:.{decimals}f}"
    return val_str


def fmt_errbar_sci_decimals(
    v: float,
    lo: float,
    hi: float,
) -> str:
    """
  Put v in convenient scientific basis, pick decimals from largest uncertainty (1 sig digit),
  but cap at 2 decimals. If k==0, no outer parentheses; if k!=0, wrap in parentheses and add x10^k.
  Clip uncertainties so they never round to 0.0.
  """
    m, k = sci_basis(v)

    # Scale uncertainties into the same basis
    scale = 10**k
    lo_s = lo / scale
    hi_s = hi / scale

    # Decide decimals from the max error (one sig digit), then cap at 2 dp
    max_e = max(lo_s, hi_s)
    dec_rule = decimals_from_max_err(max_e)
    decimals = min(dec_rule, 2)

    m_str = round_fixed(m, decimals)
    lo_str = round_fixed(lo_s, decimals)
    hi_str = round_fixed(hi_s, decimals)

    lo_str = clip_uncertainty(lo_str, decimals)
    hi_str = clip_uncertainty(hi_str, decimals)

    if k == 0:
        return f"${m_str}_{{-{lo_str}}}^{{+{hi_str}}}$"
    else:
        core = f"\\left({m_str}_{{-{lo_str}}}^{{+{hi_str}}}\\right)"
        return f"${core}\\times 10^{{{k}}}$"


def fmt_from_log10_block(
    d: dict,
) -> str:
    v, lo, hi = from_log10(
        d["p50"],
        d["std_lo"],
        d["std_hi"],
    )
    return fmt_errbar_sci_decimals(
        v,
        lo,
        hi,
    )


def fmt_from_linear_block(
    d: dict,
) -> str:
    return fmt_errbar_sci_decimals(
        d["p50"],
        d["std_lo"],
        d["std_hi"],
    )


def fmt_duration_block(
    d: dict,
) -> str:
    """
  Special formatter for nonlinear duration.
  Always report linear value with exactly 1 decimal place, no scientific basis.
  Clip uncertainties to at least 0.1.
  """
    v, lo, hi = from_log10(
        d["p50"],
        d["std_lo"],
        d["std_hi"],
    )
    v_str = f"{v:.1f}"
    lo_str = f"{lo:.1f}"
    hi_str = f"{hi:.1f}"
    if float(lo_str) == 0.0: lo_str = "0.1"
    if float(hi_str) == 0.0: hi_str = "0.1"
    return f"${v_str}_{{-{lo_str}}}^{{+{hi_str}}}$"


def sims_at_res(
    count: int,
    Nres: int,
) -> str:
    """Return 'count x Nres' if count>1, else 'Nres' (LaTeX-friendly)."""
    return f"{count}\\,$\\times$\\,{Nres}"  #if count > 1 else f"{Nres}"


def build_table(
    dataset: dict[str, dict],
) -> str:

    rows = []
    for _, suite_stats in dataset.items():
        count = int(suite_stats["input"]["count"])
        Nres = int(suite_stats["input"]["Nres"])
        nu = float(suite_stats["input"]["nu"])

        m_Mach = suite_stats["input"]["target_Mach"]
        m_Re = suite_stats["input"]["target_Re"]
        m_alpha = suite_stats["measured"]["log10_alpha_nl"]
        m_gamma = suite_stats["measured"]["log10_gamma_exp_times_t0"]
        m_tnl = suite_stats["measured"]["log10_nl_duration_normed_by_t0"]
        m_pnl = suite_stats["measured"]["p_nl"]

        mach_str = str(m_Mach)
        re_str = str(m_Re)
        nu_str = fmt_value_sci(
            nu,
            sig=2,
        )  # nu: simple scientific, 2 sig figs
        sims_str = sims_at_res(
            count,
            Nres,
        )
        gamma_str = fmt_from_log10_block(m_gamma)
        alpha_str = fmt_from_log10_block(m_alpha)
        pnl_str = fmt_from_linear_block(m_pnl)
        tnl_str = fmt_duration_block(m_tnl)

        rows.append(
            f"{mach_str} & {re_str} & {nu_str} & {sims_str} & {gamma_str} & {alpha_str} & {pnl_str} & {tnl_str} \\\\",
        )
    return "\n\n".join(rows)


##
## === MAIN PROGRAM
##


def main() -> None:
    script_dir = Path(__file__).parent
    dataset_path = (script_dir / ".." / ".." / "datasets" / "summary_v2.json").resolve()
    dataset = json_io.read_json_file_into_dict(
        dataset_path,
        verbose=False,
    )

    def sort_key(
        item: tuple[str, dict],
    ) -> tuple[float, float, int]:
        s = item[1]
        # Use target inputs for ordering, then resolution
        mach = float(s["input"]["target_Mach"])
        re_ = float(s["input"]["target_Re"])
        nres = int(s["input"]["Nres"])
        return (mach, re_, nres)

    dataset_sorted = dict(
        sorted(
            dataset.items(),
            key=sort_key,
        ),
    )
    table_tex = build_table(dataset_sorted)
    print(table_tex)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
