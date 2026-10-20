"""
viability.py -- operative viability boundary of law A (fixed a, linear
mass loss with no cap, so that m(v) reaches a at v_ext = v_on + (M-a)/alpha).

A configuration is viable if some incoming trajectory of the launch family
(r0 = 10M, E0 = 1, L0 grid, episode starting at r = 3M) reaches a point
inside the instantaneous ergosphere, outside r_+(v), before v_ext: only
there can a decay produce a negative-energy fragment (Corollary 1).
We locate alpha_crit(a) by bisection and compare with the analytic
estimate alpha_ext = (M-a)/Delta v_min, where Delta v_min is the shortest
advanced-time interval between the episode start and ergosphere entry
measured on the static (alpha=0) trajectories.
"""
import json
import numpy as np
import kvcore as K

import sys
R0, RON = 10.0, 3.0
E0 = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
L_grid = np.linspace(-4.0, 4.5, 69)
DEPTH = 1e-3     # entry depth below the ergosurface

def entries(a, alpha):
    Pst = np.array([0.0, 0.0, 1.0, a, 1.0, 0.0])
    dv = []
    for L0 in L_grid:
        pr0 = K.pr_from_shell(R0, 1.0, a, -E0, L0, 1.0, 'in')
        if pr0 is None:
            continue
        y0 = np.array([0.0, R0, 0.0, -E0, pr0, L0])
        yon, _, st0, _, _, _ = K.integrate(y0, Pst, 1e5, 1e4, 2, RON)
        if st0 != K.EVENT:
            continue
        P = np.array([0.0, alpha, 1.0 - a + 0.5, a, 1.0, yon[0]])   # uncapped: reaches m = a
        ys, _, st, _, _, _ = K.integrate(yon, P, 1e5, 1e4, 3, -DEPTH)
        if st != K.EVENT:
            continue
        m, aa, _, _ = K.mass_state(ys[0], P)
        if m < aa or ys[1] <= K.r_plus(m, aa):
            continue
        dv.append(ys[0] - yon[0])
    return dv

out = []
for a in ((0.8, 0.85, 0.9, 0.95, 0.98, 0.99) if E0 == 1.0 else (0.8, 0.9, 0.95, 0.99)):
    dv0 = entries(a, 0.0)
    dvmin = min(dv0)
    lo, hi = 1e-5, 1.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if entries(a, mid):
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0005:
            break
    row = dict(a=a, dv_min=dvmin, alpha_crit=lo, alpha_pred=(1 - a) / dvmin)
    out.append(row)
    print(row)
json.dump(out, open('../data/viability.json' if E0 == 1.0 else f'../data/viability_E{E0}.json', 'w'), indent=1)
