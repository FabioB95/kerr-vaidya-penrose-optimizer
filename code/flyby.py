"""
flyby.py -- pure energy transport (no decay) for scattering geodesics.

A unit-mass particle with E0 = 1.1 starts at r = 50 M (v = 0), falls in,
passes a pericentre of a few M and escapes. The mass-loss episode (Delta m
= 0.05) starts when the particle crosses r = 6 M inbound. We measure
Delta E = E_inf - E0 versus alpha, for a = 0 (Vaidya: no ergosphere outside
the horizon, so no Penrose channel at all) and a = 0.9 (laws A and B), and
compare with
  * the first-order (small-alpha) prediction obtained by integrating the
    exact rate dE/dtau along the STATIC geodesic over the episode, and
  * the impulsive (large-alpha) limit, Delta E -> Delta m * [dE/dm]_on with
    dE/dm = p_r^2/(r u^v) (law A) evaluated at the episode start.
"""
import json
import numpy as np
import kvcore as K

E0, R0, RON, DM = 1.1, 50.0, 6.0, 0.05
alphas = np.logspace(-5, 0.5, 23)
out = {}
for law, a, L0 in [(0, 0.0, 4.75), (0, 0.9, 3.9), (1, 0.9, 3.9), (0, 0.9, 3.0), (1, 0.9, 3.0)]:
    key = f"law{'AB'[law]}_a{a}_L{L0}"
    Pst = np.array([float(law), 0.0, DM, a, 1.0, 0.0])
    pr0 = K.pr_from_shell(R0, 1.0, a, -E0, L0, 1.0, 'in')
    y0 = np.array([0.0, R0, 0.0, -E0, pr0, L0])
    yon, _, st, _, _, _ = K.integrate(y0, Pst, 1e6, 1e9, 2, RON)
    assert st == K.EVENT, (key, st)
    von = yon[0]
    # static continuation (reference trajectory for first-order estimate)
    yS, _, stS, rminS, tr, nt = K.integrate(yon, Pst, 1e8, 3e4, 0, 0.0, record=3_000_000)
    assert stS == K.ESCAPED, (key, stS)
    tr = tr[:nt]
    Gi = [K.ginv_eq(tr[i, 1], 1.0, a) for i in range(nt)]
    uv = np.array([Gi[i][0] @ tr[i, 3:6] for i in range(nt)])
    pr = tr[:, 4]; r = tr[:, 1]; v = tr[:, 0]; pp = tr[:, 5]; pv = tr[:, 3]
    # dE/dm along the static path (per unit mass lost, at that v):
    # law A: p_r^2/(r u^v); law B adds -chi (p_v+p_r)(a(p_v+p_r)+p_phi)/(r^2 u^v)
    dEdm = pr**2 / (r * uv)
    if law == 1:
        s = pv + pr
        dEdm = dEdm - a * s * (a * s + pp) / (r**2 * uv)
    rows = []
    for al in alphas:
        P = np.array([float(law), al, DM, a, 1.0, von])
        yf, _, stf, rmin, _, _ = K.integrate(yon, P, 1e7, 400.0, 0, 0.0)
        dE = -yf[3] - E0 if stf == K.ESCAPED else np.nan
        # first-order: Delta E ~ alpha * int dEdm dv over v in [von, von+DM/al]
        vend = von + DM / al
        mask = v <= vend
        dE1 = al * np.trapezoid(dEdm[mask], v[mask]) if mask.sum() > 1 else 0.0
        if mask.sum() < len(v) and mask.sum() > 0:   # partial last segment
            i = mask.sum()
            frac = (vend - v[i-1]) / (v[i] - v[i-1])
            dE1 += al * 0.5 * (dEdm[i-1] + dEdm[i-1] + frac*(dEdm[i]-dEdm[i-1])) * (vend - v[i-1])
        rows.append(dict(alpha=al, dE=dE, status=int(stf), dE_first_order=dE1, rmin=rmin))
    out[key] = dict(law=law, a=a, L0=L0, v_on=von, rmin_static=rminS,
                    dE_impulsive=DM * dEdm[0], dEdm_total=float(np.trapezoid(dEdm, v)),
                    rows=rows)
    print(key, "rmin", round(rminS, 3), "impulsive", DM * dEdm[0])
    for rw in rows[::3]:
        print(f"   alpha={rw['alpha']:.2e} dE={rw['dE']:.3e} first-order={rw['dE_first_order']:.3e}")
json.dump(out, open('../data/flyby.json', 'w'), indent=1, default=float)
