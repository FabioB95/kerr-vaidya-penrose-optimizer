"""run_convergence_full.py -- convergenza per Appendice B2, sia statico che dinamico"""
import json
import numpy as np
import sys
from pathlib import Path

# Robust path handling (works from any working directory)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from pipeline_v2 import run_case_v2

# Find sweep_results.json next to the script or one level up
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

def clean(d):
    if d is None:
        return None
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in d.items()}

print("--- Static convergence (a/M=0.9) ---")
for n in [8, 12, 16, 20, 24, 32]:
    E_grid = np.linspace(0.5, 1.05, n)
    L_grid = np.linspace(-6.0, 6.0, int(n * 1.5))
    res = run_case_v2(a=0.9, alpha=0.0, drmin=0.001, E_grid=E_grid, L_grid=L_grid)
    key = f"convergence_static_n{n}"
    if res:
        print(f"n={n}: dE={res['dE']:.5f} eta={res['eta']*100:.2f}%")
    OUT[key] = clean(res)

print("\n--- Dynamic convergence (a/M=0.9, alpha=5e-4) ---")
for n in [8, 12, 16, 20, 24, 32]:
    E_grid = np.linspace(0.5, 1.05, n)
    L_grid = np.linspace(-8.0, 8.0, int(n * 1.5))
    res = run_case_v2(a=0.9, alpha=5e-4, drmin=0.001, E_grid=E_grid,
                       L_grid=L_grid, tau_max=400.0)
    key = f"convergence_dynamic_n{n}"
    if res:
        print(f"n={n}: dE={res['dE']:.5f} eta={res['eta']*100:.2f}%")
    OUT[key] = clean(res)

with open(json_path, "w") as f:
    json.dump(OUT, f, indent=2, default=str)

print(f"\nUpdated {json_path}")