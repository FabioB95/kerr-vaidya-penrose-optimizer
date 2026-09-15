"""
full_split.py  --  versione rigorosa (fotone uscente + frammento 2 timelike generico)
"""
import numpy as np
from scipy.optimize import brentq
import geodesics as kv


def photon_pr_roots(E, L, ginv0):
    pv, pphi = -E, L
    A = ginv0[1, 1]
    B = 2 * (ginv0[1, 0] * pv + ginv0[1, 2] * pphi)
    C = ginv0[0, 0]*pv**2 + 2*ginv0[0, 2]*pv*pphi + ginv0[2, 2]*pphi**2
    disc = B**2 - 4*A*C
    with np.errstate(invalid='ignore'):
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
    r1 = (-B - sq) / (2*A)
    r2 = (-B + sq) / (2*A)
    return np.minimum(r1, r2), np.maximum(r1, r2)


def compute_mu2(E, L, pr, ginv0):
    p = np.array([-E, pr, L], dtype=float)
    return float( -(p @ (ginv0 @ p)) )


def irreducible_mass(M, a_spin):
    if M**2 < a_spin**2:
        return None
    rp = M + np.sqrt(M**2 - a_spin**2)
    return np.sqrt(rp**2 + a_spin**2) / 2.0


def _exact_pr_photon(E, L, ginv0, branch):
    lo, hi = photon_pr_roots(np.asarray(E), np.asarray(L), ginv0)
    return float(lo if branch == 'lo' else hi)


def drdtau_sign(E, L, pr, ginv0):
    p = np.array([-E, pr, L], dtype=float)
    return float( (ginv0 @ p)[1] )


def _collect_candidates(E3g, L3g, pr3_branch, E1, L1, pr1, ginv0,
                        Mirr_old, Ja, m, branch3_name):
    cands = []
    n_E3, n_L3 = pr3_branch.shape

    for j in range(n_L3):
        col = pr3_branch[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        idx_valid = np.where(valid)[0]
        L3_val = L3g[j]

        for k in range(len(idx_valid) - 1):
            i0 = idx_valid[k]
            i1 = idx_valid[k + 1]
            if i1 != i0 + 1:
                continue

            # campiona alcuni E3 nell'intervallo
            for E3 in np.linspace(E3g[i0], E3g[i1], 6):
                try:
                    pr3 = _exact_pr_photon(E3, L3_val, ginv0, branch3_name)
                except Exception:
                    continue

                E2 = E1 - E3
                L2 = L1 - L3_val
                pr2 = pr1 - pr3

                if E2 >= 0:
                    continue

                mu2 = compute_mu2(E2, L2, pr2, ginv0)
                if mu2 <= 0:
                    continue

                M_new = m + E2
                J_new = Ja + L2
                if M_new <= 0:
                    continue
                a_new = J_new / M_new
                if M_new**2 < a_new**2:
                    continue

                rp_new = M_new + np.sqrt(M_new**2 - a_new**2)
                Mirr_new = np.sqrt(rp_new**2 + a_new**2) / 2.0
                if Mirr_new < Mirr_old - 1e-9:
                    continue

                eta = (E3 - E1) / E1
                cands.append(dict(
                    eta=eta, E2=E2, L2=L2, pr2=pr2,
                    E3=E3, L3=L3_val, pr3=pr3, mu2=mu2
                ))

    cands.sort(key=lambda c: c['eta'], reverse=True)
    return cands


def all_valid_splits(r, m, a, E1, L1, pr1,
                     n_L3=350, n_E3=280,
                     L3_span=(-10, 10), E3_span=(-2.0, 4.0)):
    ginv0 = np.array(kv.ginv_f(r, a, m, 0.0), dtype=float)
    Mirr_old = irreducible_mass(m, a)
    if Mirr_old is None:
        return []
    Ja = a * m

    L3g = np.linspace(L1 + L3_span[0], L1 + L3_span[1], n_L3)
    E3g = np.linspace(E1 + E3_span[0], E1 + E3_span[1], n_E3)
    E3grid, L3grid = np.meshgrid(E3g, L3g, indexing='ij')

    pr3_lo, pr3_hi = photon_pr_roots(E3grid, L3grid, ginv0)

    all_cands = []
    for pr3_b, b3 in ((pr3_lo, 'lo'), (pr3_hi, 'hi')):
        cands = _collect_candidates(
            E3g, L3g, pr3_b, E1, L1, pr1, ginv0,
            Mirr_old, Ja, m, b3
        )
        all_cands.extend(cands)

    all_cands.sort(key=lambda c: c['eta'], reverse=True)
    return all_cands


def all_valid_escaping_splits(r, m, a, E1, L1, pr1,
                               n_L3=350, n_E3=280,
                               L3_span=(-10, 10), E3_span=(-2.0, 4.0)):
    ginv0 = np.array(kv.ginv_f(r, a, m, 0.0), dtype=float)
    cands = all_valid_splits(r, m, a, E1, L1, pr1, n_L3, n_E3, L3_span, E3_span)
    return [c for c in cands if drdtau_sign(c['E3'], c['L3'], c['pr3'], ginv0) > 0]