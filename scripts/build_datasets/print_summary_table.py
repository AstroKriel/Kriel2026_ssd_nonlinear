import numpy
from pathlib import Path
from collections import defaultdict
from jormi.ww_io import io_manager, json_files

def format_scientific(
    val     : float,
    sigfigs : int = 1,
  ) -> str:
  if val == 0: return "$0$"
  exp = int(numpy.floor(numpy.log10(abs(val))))
  coeff = round(val / 10**exp, sigfigs)
  if exp == 0: return f"${coeff}$"
  return f"${coeff}\\times10^{{{exp}}}$"

def main():
  data_dir = Path("/Users/necoturb/Documents/Codes/asgard/mimir/kriel_2025_ssd_nl/datasets_v2")
  sim_dirs = [
    sim_directory
    for sim_directory in sorted(data_dir.glob("Mach*Re*Pm1Nres*"))
    if io_manager.does_file_exist(
      directory   = sim_directory,
      file_name   = "dataset.json",
      raise_error = False
    )
  ]
  sim_set = defaultdict(lambda: defaultdict(int))
  for sim_dir in sim_dirs:
    data_filepath = sim_dir / "dataset.json"
    sim_data = json_files.read_json_file_into_dict(data_filepath, verbose=False)
    ell_0 = 0.5
    Mach = sim_data["plasma_params"]["target_Mach"]
    Re   = sim_data["plasma_params"]["target_Re"]
    nu   = ell_0 * Mach / Re
    Nres = sim_data["plasma_params"]["resolution"]
    plasma_params = (Mach, Re, nu)
    sim_set[plasma_params][Nres] += 1
  for (Mach, Re, nu), res_counts in sorted(sim_set.items()):
    runs_str = ", ".join(
      f"${count} \\times {Nres}^3$"
      if count > 0 else
      f"${Nres}^3$"
      for Nres, count in sorted(res_counts.items())
    )
    line = f"{Mach:.2f} & {Re} & {format_scientific(nu)} & {runs_str} \\\\"
    print(line)

if __name__ == "__main__":
  main()

## .