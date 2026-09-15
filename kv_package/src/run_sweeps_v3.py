"""run_sweeps_v3.py -- sweep completo col metodo a conservazione totale.
Scrive su sweep_results_v3.json (file separato, non sovrascrive il v2)."""
import json
import time
import numpy as np
import sys
sys.path.insert(0, '.')
from pipeline_v3 import run_case_v3

OUT = {}


def clean(d):
    if d is None:
        return None
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in d.items()}


def run_and_record(key, **kwargs):
    t0 = time.time()
    res = run_case_v3(**kwargs)
    dt = time.time() - t0
    if res:
        print(f"{key}: dE={res['dE']:.5f} eta={res['eta']*100:.2f}% "
              f"margin={res['margin']:.4f} ({dt:.1f}s)")
    else:
        print(f"{key}: NO VALID SPLIT ({dt:.1f}s)")
    OUT[key] = clean(res)
    return res


if __name__ == "__main__":
    # --- 1. Static spin sweep ---
    for a in [0.80, 0.90, 0.95, 0.98, 0.99]:
        run_and_record(f"static_a{a}", a=a, alpha=0.0, drmin=0.001)

    # --- 2. Dynamic spin sweep ---
    dyn_E_grid = np.linspace(0.5, 1.05, 24)
    dyn_L_grid = np.linspace(-8.0, 8.0, 36)
    for a in [0.80, 0.90, 0.95, 0.98, 0.99]:
        run_and_record(f"dynamic_a{a}_alpha5e-4", a=a, alpha=5e-4, drmin=0.001,
                        E_grid=dyn_E_grid, L_grid=dyn_L_grid, tau_max=400.0)

    # --- 3. Alpha sweep at a/M=0.9 ---
    for alpha in [0.0, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3]:
        run_and_record(f"alphasweep_a0.9_alpha{alpha}", a=0.9, alpha=alpha,
                        drmin=0.001, E_grid=dyn_E_grid, L_grid=dyn_L_grid,
                        tau_max=400.0)

    # --- 4. Phase boundary ---
    def alpha_crit_binary_search(a, alpha_min=1e-5, alpha_max=0.01, tol=2e-4, max_iter=16):
        lo, hi = alpha_min, alpha_max
        tries = 0
        while run_case_v3(a=a, alpha=hi, drmin=0.001) is not None and tries < 6:
            print(f"  still viable at alpha={hi}, widening...")
            hi *= 2
            tries += 1
        if run_case_v3(a=a, alpha=lo, drmin=0.001) is None:
            print(f"  WARNING: not viable even at alpha_min={lo}")
            return None
        for _ in range(max_iter):
            if hi - lo < tol:
                break
            mid = 0.5 * (lo + hi)
            if run_case_v3(a=a, alpha=mid, drmin=0.001) is not None:
                lo = mid
            else:
                hi = mid
        return lo

    for a in [0.90, 0.95, 0.99]:
        crit = alpha_crit_binary_search(a)
        print(f"alpha_crit({a}) = {crit}")
        OUT[f"alpha_crit_a{a}"] = crit

    # --- 5. Convergence (static + dynamic) ---
    for n in [8, 12, 16, 20, 24, 32]:
        E_grid = np.linspace(0.5, 1.05, n)
        L_grid = np.linspace(-6.0, 6.0, int(n * 1.5))
        run_and_record(f"convergence_static_n{n}", a=0.9, alpha=0.0, drmin=0.001,
                        E_grid=E_grid, L_grid=L_grid)
    for n in [8, 12, 16, 20, 24, 32]:
        E_grid = np.linspace(0.5, 1.05, n)
        L_grid = np.linspace(-8.0, 8.0, int(n * 1.5))
        run_and_record(f"convergence_dynamic_n{n}", a=0.9, alpha=5e-4, drmin=0.001,
                        E_grid=E_grid, L_grid=L_grid, tau_max=400.0)

    with open("sweep_results_v3.json", "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print("\nSaved sweep_results_v3.json")