"""
Targeted rerun of just the two items that needed fixing from the first
sweep -- no need to redo the static sweep, alpha sweep, or convergence
table, those looked fine. Merges into the existing sweep_results.json.
"""
import json
import numpy as np
import sys
sys.path.insert(0, '.')
from pipeline_v2 import run_case_v2

with open("sweep_results.json") as f:
    OUT = json.load(f)


def clean(d):
    if d is None:
        return None
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in d.items()}


# --- Dynamic case, wider grid + longer integration (was finding worse
#     margins than the static case, and even higher dE for high spin -- see
#     chat: this turned out to be a genuine feature of the a-fixed Kerr-
#     Vaidya convention (a/m(v) drifts toward extremality as m decreases),
#     not a bug, but we still want the best achievable margin/dE at each
#     point rather than an under-resolved search) ---
dyn_E_grid = np.linspace(0.5, 1.05, 24)
dyn_L_grid = np.linspace(-8.0, 8.0, 36)
for a in [0.80, 0.90, 0.95, 0.98, 0.99]:
    key = f"dynamic_a{a}_alpha5e-4"
    res = run_case_v2(a=a, alpha=5e-4, drmin=0.001, E_grid=dyn_E_grid,
                       L_grid=dyn_L_grid, tau_max=400.0)
    if res:
        print(f"{key}: dE={res['dE']:.5f} eta={res['eta']*100:.2f}% "
              f"margin={res['margin']:.4f}  m_split={res['m_split']:.4f}  "
              f"a/m_split={a/res['m_split']:.4f} (nominal a/M={a})")
    OUT[key] = clean(res)

# --- Phase boundary, widened alpha_max until we actually bracket the true
#     cutoff instead of saturating at the edge of the search range ---
def alpha_crit_binary_search(a, alpha_min=1e-5, alpha_max=0.3, tol=2e-4, max_iter=16):
    lo, hi = alpha_min, alpha_max
    tries = 0
    while run_case_v2(a=a, alpha=hi, drmin=0.001) is not None and tries < 6:
        print(f"  still viable at alpha={hi}, widening...")
        hi *= 2
        tries += 1
    if run_case_v2(a=a, alpha=lo, drmin=0.001) is None:
        print(f"  WARNING: not viable even at alpha_min={lo} -- check this spin separately")
        return None
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if run_case_v2(a=a, alpha=mid, drmin=0.001) is not None:
            lo = mid
        else:
            hi = mid
    return lo


for a in [0.90, 0.95]:
    crit = alpha_crit_binary_search(a)
    print(f"alpha_crit({a}) = {crit}")
    OUT[f"alpha_crit_a{a}"] = crit

with open("sweep_results.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nUpdated sweep_results.json")
