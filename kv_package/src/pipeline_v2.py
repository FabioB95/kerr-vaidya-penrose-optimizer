import numpy as np
import geodesics as kv
from pipeline import (pr_from_mass_shell, integrate, verify_escape, m_of_v,
                       R0, DRMIN_DEFAULT)
from split_extraction import max_extraction


def run_case_v2(a, alpha, m0=1.0, drmin=DRMIN_DEFAULT, E_grid=None, L_grid=None,
                 tau_max=250.0, verbose=False):
    """For each incoming (E,L) trajectory: integrate, find the deepest safe
    point, compute the geometry-only maximum extraction there (dE, via the
    validated photon-cone + area-theorem method), and KEEP it only if the
    resulting escaping fragment 3 (E3=E1-E2, L3=L-L2) genuinely reaches
    R_ESC. Track the best (highest dE) escape-verified candidate."""
    if E_grid is None:
        E_grid = np.linspace(0.5, 1.05, 16)
    if L_grid is None:
        L_grid = np.linspace(-6.0, 6.0, 24)

    kv.m0_global = m0
    r_plus_0 = kv.r_plus(m0, a)
    best = None

    for E in E_grid:
        for L in L_grid:
            pr0 = pr_from_mass_shell(R0, a, m0, E, L, 'in')
            if pr0 is None:
                continue
            y0 = [0.0, R0, 0.0, -E, pr0, L]
            sol = integrate(y0, a, alpha, tau_max, r_plus_0)
            r_track, v_track, phi_track = sol.y[1], sol.y[0], sol.y[2]
            m_track = m_of_v(v_track, m0, alpha)
            rs_track = kv.r_s(m_track)
            rplus_track = np.array([kv.r_plus(mm, a) for mm in m_track])
            inside = r_track < rs_track
            if not np.any(inside):
                continue
            margin = r_track - rplus_track
            safe_inside = inside & (margin >= drmin)
            if not np.any(safe_inside):
                continue
            masked = np.where(safe_inside, r_track, np.inf)
            idx = np.argmin(masked)
            r_s_, v_s_, phi_s_, m_s_ = r_track[idx], v_track[idx], phi_track[idx], m_track[idx]

            ext = max_extraction(r_s_, m_s_, a)
            if ext is None or ext['dE'] <= 0:
                continue
            if best is not None and ext['dE'] <= best['dE']:
                continue  # cheap pre-filter before the expensive escape check

            E3 = E - ext['E2']
            L3 = L - ext['L2']
            ok = verify_escape(r_s_, v_s_, phi_s_, E3, L3, a, alpha, m0)
            if not ok:
                continue
            best = dict(dE=ext['dE'], eta=ext['dE'] / E, E1=E, L1=L,
                        E2=ext['E2'], L2=ext['L2'], E3=E3, L3=L3,
                        r_split=r_s_, v_split=v_s_, phi_split=phi_s_, m_split=m_s_,
                        margin=r_s_ - kv.r_plus(m_s_, a))
            if verbose:
                print(f"  new best: E={E:.3f} L={L:.3f} -> dE={ext['dE']:.5f} "
                      f"margin={best['margin']:.4f}")
    return best


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = run_case_v2(0.9, 0.0, drmin=0.001, verbose=True)
    print("\nFinal:", res)
    print("time:", time.time() - t0, "s")
