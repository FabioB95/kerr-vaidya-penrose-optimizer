"""rerun_alphasweep_consistent.py"""
import json
import numpy as np
import sys
from pathlib import Path

# Make sure we can import pipeline_v2 no matter where we launch the script from
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))          # src/ itself
sys.path.insert(0, str(SCRIPT_DIR.parent))   # package root (in case pipeline_v2 lives there)

from pipeline_v2 import run_case_v2

# Locate sweep_results.json next to this script (or one level up if you prefer)
json_path = SCRIPT_DIR / "sweep_results.json"
if not json_path.exists():
    json_path = SCRIPT_DIR.parent / "sweep_results.json"
if not json_path.exists():
    raise FileNotFoundError(
        f"Could not find sweep_results.json.\n"
        f"Tried:\n  {SCRIPT_DIR / 'sweep_results.json'}\n  {SCRIPT_DIR.parent / 'sweep_results.json'}"
    )

with open(json_path) as f:
    OUT = json.load(f)

dyn_E_grid = np.linspace(0.5, 1.05, 24)
dyn_L_grid = np.linspace(-8.0, 8.0, 36)

for alpha in [0.0, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3]:
    key = f"alphasweep_a0.9_alpha{alpha}"
    res = run_case_v2(
        a=0.9,
        alpha=alpha,
        drmin=0.001,
        E_grid=dyn_E_grid,
        L_grid=dyn_L_grid,
        tau_max=400.0,
    )
    if res:
        print(f"{key}: dE={res['dE']:.5f} eta={res['eta']*100:.2f}%")
        OUT[key] = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in res.items()
        }
    else:
        OUT[key] = None

with open(json_path, "w") as f:
    json.dump(OUT, f, indent=2, default=str)

print(f"Updated {json_path}")
print("Now re-run figures_final.py to regenerate fig_alpha_sweep.pdf")