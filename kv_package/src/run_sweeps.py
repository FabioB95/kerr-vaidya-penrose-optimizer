"""
Full data generation using the VALIDATED pipeline (pipeline_v2.run_case_v2):
  - stage 1: alpha-aware geodesic integration from r0=6M, grid over (E,L)
  - stage 2: validated max-extraction at the deepest safe point (photon-cone
    parametrization + Wald/Christodoulou area theorem saturation)
  - escape of the resulting fragment 3 is VERIFIED by genuine ODE integration
    to r=20M before a candidate is accepted

Methodology note for the paper: dE (=|E2|) is the physically bounded
quantity, fixed by the local geometry (r_split, m_split, a) alone -- NOT by
E1. eta = dE/E1 can be inflated by choosing small E1 (bound-orbit initial
conditions), exactly as discussed in the original Sec. 4.1, but dE itself
is the number that should anchor the physics discussion (Table 3, Fig. 5).

Runtime: each run_case_v2 call scans a (E,L) grid (default 16x24=384
trajectories) with a full ODE integration per trajectory, plus one
escape-verification integration per accepted candidate. Expect ~5-60s per
call depending on grid size and how close to the horizon convergence
happens. The full sweep below (spins x 2 (static/dynamic) + alpha sweep +
phase boundary) is on the order of 30-60 minutes total on a normal laptop.
Increase E_grid/L_grid resolution for a convergence check (recommended:
report a convergence table analogous to the original Table B2, but now
for BOTH static and dynamic cases as the referee requested).
"""
import json
import time
import numpy as np
import sys
sys.path.insert(0, '.')
from pipeline_v2 import run_case_v2

OUT = {}


def clean(d):
    if d is None:
        return None
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in d.items()}


def run_and_record(key, **kwargs):
    t0 = time.time()
    res = run_case_v2(**kwargs)
    dt = time.time() - t0
    if res:
        print(f"{key}: dE={res['dE']:.5f} eta={res['eta']*100:.2f}% "
              f"margin={res['margin']:.4f} ({dt:.1f}s)")
    else:
        print(f"{key}: NO VALID SPLIT ({dt:.1f}s)")
    OUT[key] = clean(res)
    return res


if __name__ == "__main__":
    # --- 1. Static Kerr benchmark, spin sweep (Table 3 static column + Fig 5b) ---
    for a in [0.80, 0.90, 0.95, 0.98, 0.99]:
        run_and_record(f"static_a{a}", a=a, alpha=0.0, drmin=0.001)

    # --- 2. Dynamic case, same spins, fixed alpha=5e-4 (Table 3 dynamic column) ---
    # NOTE: the first run found LARGER (worse) margins here than in the static
    # case, and even higher dE for some spins -- physically backwards (a
    # shrinking ergosphere should make extraction harder, not easier). Using
    # a wider/finer grid and longer integration time to check whether this
    # was a grid-resolution artifact before trusting these numbers.
    dyn_E_grid = np.linspace(0.5, 1.05, 24)
    dyn_L_grid = np.linspace(-8.0, 8.0, 36)
    for a in [0.80, 0.90, 0.95, 0.98, 0.99]:
        run_and_record(f"dynamic_a{a}_alpha5e-4", a=a, alpha=5e-4, drmin=0.001,
                        E_grid=dyn_E_grid, L_grid=dyn_L_grid, tau_max=400.0)

    # --- 3. Alpha sweep at a/M=0.9 (Fig. 5a) ---
    for alpha in [0.0, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3]:
        run_and_record(f"alphasweep_a0.9_alpha{alpha}", a=0.9, alpha=alpha, drmin=0.001)

    # --- 4. Phase boundary: binary search for alpha_crit (Fig. 6) ---
    # NOTE: widened alpha_max substantially -- the first run saturated at the
    # old alpha_max=0.02 without finding a true cutoff, meaning the process
    # was still viable there. We need to actually bracket the true boundary.
    def alpha_crit_binary_search(a, alpha_min=1e-5, alpha_max=0.3, tol=2e-4, max_iter=14):
        lo, hi = alpha_min, alpha_max
        if run_case_v2(a=a, alpha=hi, drmin=0.001) is not None:
            print(f"  WARNING: still viable at alpha_max={hi} -- widen further")
            return hi
        if run_case_v2(a=a, alpha=lo, drmin=0.001) is None:
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

    # --- 5. Convergence check (grid resolution) at the static a/M=0.9 benchmark ---
    for n in [8, 12, 16, 20, 24]:
        E_grid = np.linspace(0.5, 1.05, n)
        L_grid = np.linspace(-6.0, 6.0, int(n * 1.5))
        run_and_record(f"convergence_n{n}", a=0.9, alpha=0.0, drmin=0.001,
                        E_grid=E_grid, L_grid=L_grid)

    with open("sweep_results.json", "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print("\nSaved sweep_results.json -- send this back for figure/table regeneration.")
