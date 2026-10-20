"""
study.py -- Penrose decays with full four-momentum conservation in the
equatorial Kerr--Vaidya spacetime; energy-budget decomposition.

For every (law, spin, alpha):
  incoming particle: unit mass, starts at r0 = 10 M, v = 0, energy E0 = 1,
  angular momentum L0 on a grid, ingoing.
  split event: first crossing of r = r_+(v) + delta  (delta on a grid,
  event must lie inside the instantaneous ergosphere r < 2 m(v)).
  decay: beta = 1 (two photons) by default, back-to-back along n(psi).
  requirements: E2 < 0 at the split, fragment 2 captured (crosses the
  final event horizon), fragment 3 escapes to r_far with v > v_end.
  objective: net gain G = E3(infinity) - E0.
Decomposition: G = (E1s - E0) + (-E2s) + (E3inf - E3s)
               transport-in   local      transport-out
"""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import kvcore as K

R0 = 10.0
E0 = 1.0
R_FAR = 200.0
R_ON = 3.0      # the mass-loss episode starts when the incoming particle crosses r = 3M


def dm_for(law, spin, dm_max=0.05):
    if law == 0:                      # fixed a: stay away from extremality
        return min(dm_max, 0.9 * (1.0 - spin))
    return dm_max


def evaluate_full(law, spin, alpha, dm, beta, L0, delta, psi):
    """Full evaluation of one configuration (used by the continuous refinement)."""
    if delta <= 0:
        return None
    Pst = np.array([float(law), 0.0, dm, spin, 1.0, 0.0])
    pr0 = K.pr_from_shell(R0, 1.0, spin, -E0, L0, 1.0, 'in')
    if pr0 is None:
        return None
    y0 = np.array([0.0, R0, 0.0, -E0, pr0, L0])
    yon, _, st0, _, _, _ = K.integrate(y0, Pst, 1e5, R_FAR, 2, R_ON)
    if st0 != K.EVENT:
        return None
    P = np.array([float(law), alpha, dm, spin, 1.0, yon[0]])
    ys, _, st, _, _, _ = K.integrate(yon, P, 1e5, R_FAR, 1, delta)
    if st != K.EVENT:
        return None
    ms, as_, _, _ = K.mass_state(ys[0], P)
    if ys[1] >= 2 * ms:
        return None
    p2, p3 = K.decay(ys, P, beta, psi)
    E2s, E3s = -p2[0], -p3[0]
    if E2s >= 0:
        return None
    f2, _, s2, _, _, _ = K.integrate(np.concatenate([ys[:3], p2]), P, 1e6, R_FAR, 0, 0.0)
    if s2 != K.CAPTURED:
        return None
    f3, _, s3, _, _, _ = K.integrate(np.concatenate([ys[:3], p3]), P, 1e6, R_FAR, 0, 0.0)
    if s3 != K.ESCAPED:
        return None
    E1s = -ys[3]
    gvv = 2 * ms / ys[1] - 1
    bnd = 0.5 * (beta * np.sqrt(1 + gvv / E1s**2) - 1)
    E3inf = -f3[3]
    return dict(L0=L0, delta=delta, psi=psi, r_s=ys[1], v_s=ys[0], m_s=ms, a_s=as_,
                E1s=E1s, E2s=E2s, E3s=E3s, E3inf=E3inf, E2hor=-f2[3], G=E3inf - E0,
                T_in=E1s - E0, loc=-E2s, T_out=E3inf - E3s, eta_loc=-E2s / E1s,
                bound=bnd, v_on=P[5])


def run_case(args):
    law, spin, alpha, beta, dm = args[:5]
    resf = args[5] if len(args) > 5 else 1
    Pst = np.array([float(law), 0.0, dm, spin, 1.0, 0.0])
    a0 = spin
    L_grid = np.linspace(-4.0, 4.5, 68 * resf + 1)
    d_grid = np.array([0.001, 0.003, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.28, 0.36, 0.45, 0.55, 0.7])
    if resf > 1:   # refined split grid: geometric interpolation between the base nodes
        d_grid = np.unique(np.concatenate([d_grid, np.sqrt(d_grid[1:] * d_grid[:-1])]))
    psis = np.linspace(-np.pi, np.pi, 288 * resf + 1)[:-1]
    best = None
    best_loc = None
    allrec = []
    nvalid = 0
    bound_viol = 0.0
    for L0 in L_grid:
        pr0 = K.pr_from_shell(R0, 1.0, a0, -E0, L0, 1.0, 'in')
        if pr0 is None:
            continue
        y0 = np.array([0.0, R0, 0.0, -E0, pr0, L0])
        # static approach until r = R_ON, where the mass-loss episode starts
        yon, _, st0, _, _, _ = K.integrate(y0, Pst, 1e5, R_FAR, 2, R_ON)
        if st0 != K.EVENT:
            continue
        P = np.array([float(law), alpha, dm, spin, 1.0, yon[0]])
        for d in d_grid:
            ys, tau, st, rmin, _, _ = K.integrate(yon, P, 1e5, R_FAR, 1, d)
            if st != K.EVENT:
                continue
            ms, as_, _, _ = K.mass_state(ys[0], P)
            if ys[1] >= 2 * ms:          # not inside ergosphere
                continue
            E1s = -ys[3]
            gvv = 2 * ms / ys[1] - 1
            bnd = 0.5 * (beta * np.sqrt(1 + gvv / E1s**2) - 1)
            # coarse psi scan, then refine around the best valid psi
            def evaluate(psi):
                p2, p3 = K.decay(ys, P, beta, psi)
                E2s, E3s = -p2[0], -p3[0]
                if E2s >= 0:
                    return None
                y2 = np.concatenate([ys[:3], p2])
                f2, _, s2, _, _, _ = K.integrate(y2, P, 1e6, R_FAR, 0, 0.0)
                if s2 != K.CAPTURED:
                    return None
                y3 = np.concatenate([ys[:3], p3])
                f3, _, s3, _, _, _ = K.integrate(y3, P, 1e6, R_FAR, 0, 0.0)
                if s3 != K.ESCAPED:
                    return None
                E3inf = -f3[3]
                return dict(psi=psi, E2s=E2s, E3s=E3s, E3inf=E3inf,
                            E2hor=-f2[3])
            res = [evaluate(p) for p in psis]
            ok = [x for x in res if x is not None]
            if not ok:
                continue
            top = max(ok, key=lambda x: x['E3inf'])
            fine = np.linspace(top['psi'] - 0.05, top['psi'] + 0.05, 21)
            ok += [x for x in (evaluate(p) for p in fine) if x is not None]
            for x in ok:
                nvalid += 1
                eta_loc = -x['E2s'] / E1s
                bound_viol = max(bound_viol, eta_loc - bnd)
                rec = dict(L0=L0, delta=d, psi=x['psi'], r_s=ys[1], v_s=ys[0],
                           m_s=ms, a_s=as_, E1s=E1s, E2s=x['E2s'], E3s=x['E3s'],
                           E3inf=x['E3inf'], E2hor=x['E2hor'],
                           G=x['E3inf'] - E0, T_in=E1s - E0, loc=-x['E2s'],
                           T_out=x['E3inf'] - x['E3s'], eta_loc=eta_loc,
                           bound=bnd, v_on=P[5])
                allrec.append(rec)
                if best is None or rec['G'] > best['G']:
                    best = rec
                if best_loc is None or rec['eta_loc'] > best_loc['eta_loc']:
                    best_loc = rec
    # ---- continuous refinement of the best grid optima (Nelder--Mead) ----
    if best is not None:
        starts = sorted(allrec, key=lambda x: -x['G'])
        seeds = []
        for x in starts:
            if all(abs(x['L0'] - s_['L0']) > 0.2 or abs(np.log(x['delta'] / s_['delta'])) > 0.5
                   for s_ in seeds):
                seeds.append(x)
            if len(seeds) >= 5:
                break
        dL = L_grid[1] - L_grid[0]
        for sd in seeds:
            def f(x):
                r_ = evaluate_full(law, spin, alpha, dm, beta, x[0], np.exp(x[1]), x[2])
                return 1.0 if r_ is None else -r_['G']
            x0 = np.array([sd['L0'], np.log(sd['delta']), sd['psi']])
            simplex = np.array([x0, x0 + [dL, 0, 0], x0 + [0, 0.35, 0], x0 + [0, 0, 0.04]])
            opt = minimize(f, x0, method='Nelder-Mead',
                           options=dict(initial_simplex=simplex, xatol=1e-6, fatol=1e-8, maxfev=600))
            r_ = evaluate_full(law, spin, alpha, dm, beta, opt.x[0], np.exp(opt.x[1]), opt.x[2])
            if r_ is not None:
                bound_viol = max(bound_viol, r_['eta_loc'] - r_['bound'])
                if r_['G'] > best['G']:
                    best = r_
    return dict(law=law, spin=spin, alpha=alpha, beta=beta, dm=dm, res=resf,
                best=best, best_loc=best_loc, nvalid=nvalid,
                max_bound_violation=bound_viol)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    alphas = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    cases = []
    if which == "main":
        for law in (0, 1):
            for spin in (0.5, 0.7, 0.9, 0.99, 0.998):
                for al in alphas:
                    cases.append((law, spin, al, 1.0, dm_for(law, spin)))
    elif which == "all":
        for law in (0, 1):
            for spin in (0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.998):
                for al in alphas:
                    if law == 1 and al == 0.0:
                        continue
                    cases.append((law, spin, al, 1.0, dm_for(law, spin)))
    elif which == "spins":
        for law in (0, 1):
            for spin in (0.75, 0.8, 0.85, 0.95, 0.98):
                for al in (0.0, 1e-3, 1e-2):
                    if law == 1 and al == 0.0:
                        continue
                    cases.append((law, spin, al, 1.0, dm_for(law, spin)))
    elif which == "conv":
        for (law, spin, al) in ((0, 0.9, 0.0), (0, 0.9, 1e-2), (0, 0.9, 1e-3), (0, 0.99, 0.0), (1, 0.99, 1e-3)):
            cases.append((law, spin, al, 1.0, dm_for(law, spin), 2))
    elif which == "dmscan":
        for law in (0, 1):
            for dm in (0.01, 0.02, 0.05, 0.08):
                cases.append((law, 0.9, 1e-2, 1.0, dm))
    elif which == "dmsens":
        for (spin, dm) in ((0.9, 0.025), (0.99, 0.005)):
            for al in (1e-3, 1e-2):
                cases.append((0, spin, al, 1.0, dm))
    elif which == "beta":
        for b in (0.5, 0.8, 0.95, 1.0):
            for al in (0.0, 1e-2):
                cases.append((0, 0.9, al, b, dm_for(0, 0.9)))
    t0 = time.time()
    with Pool(2) as pool:
        results = pool.map(run_case, cases, chunksize=1)
    print("elapsed", time.time() - t0)
    with open(f"../data/study_{which}.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    for R in results:
        b = R['best']
        if b is None:
            print(R['law'], R['spin'], R['alpha'], "no valid decay")
            continue
        print(f"law={'AB'[R['law']]} a={R['spin']} al={R['alpha']:.0e} beta={R['beta']} "
              f"G={b['G']:.4f} in={b['T_in']:.4f} loc={b['loc']:.4f} out={b['T_out']:.4f} "
              f"etaloc_max={R['best_loc']['eta_loc']:.4f} (bnd {R['best_loc']['bound']:.4f}) "
              f"viol={R['max_bound_violation']:.1e} n={R['nvalid']}")
