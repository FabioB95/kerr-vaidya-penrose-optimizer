import numpy as np
from scipy.integrate import solve_ivp
import geodesics as kv
from optimizer2 import (pr_from_mass_shell, irreducible_mass, area_theorem_ok,
                         integrate, verify_escape, m_of_v, R0, DRMIN_DEFAULT)


def best_split_at_point(r, m_here, a, E1, L1, n_grid=150,
                         e3_span=(-1.5, 2.5), l3_span=(-5.0, 6.0)):
    """Vectorized scan over (E3,L3) at a single split point; returns the
    highest-eta candidate that passes BOTH the local reality condition and
    the global area theorem, or None."""
    ginv0 = np.array(kv.ginv_f(r, a, m_here, 0.0), dtype=float)
    E3g = np.linspace(E1 + e3_span[0], E1 + e3_span[1], n_grid)
    L3g = np.linspace(L1 + l3_span[0], L1 + l3_span[1], n_grid)
    E3, L3 = np.meshgrid(E3g, L3g, indexing='ij')
    E2, L2 = E1 - E3, L1 - L3

    def veff_grid(E, L):
        pv0, pph0 = -E, L
        A = ginv0[1, 1]
        B = 2*ginv0[1, 0]*pv0 + 2*ginv0[1, 2]*pph0
        C = ginv0[0, 0]*pv0**2 + 2*ginv0[0, 2]*pv0*pph0 + ginv0[2, 2]*pph0**2 + 1
        return -(B**2 - 4*A*C)

    ok_local = (veff_grid(E2, L2) <= 1e-9) & (veff_grid(E3, L3) <= 1e-9)
    if not np.any(ok_local):
        return None

    J_here = a * m_here
    Mirr_old = irreducible_mass(m_here, a)
    M_new = m_here + E2
    J_new = J_here + L2
    with np.errstate(invalid='ignore', divide='ignore'):
        a_new = J_new / M_new
        spin_ok = (M_new**2 >= a_new**2) & (M_new > 0)
        rp_new = np.where(spin_ok, M_new + np.sqrt(np.maximum(M_new**2 - a_new**2, 0)), np.nan)
        Mirr_new = np.sqrt(rp_new**2 + a_new**2) / 2
    ok_area = spin_ok & (Mirr_new >= Mirr_old - 1e-9)
    ok = ok_local & ok_area
    if not np.any(ok):
        return None

    eta = (E3 - E1) / E1
    eta_m = np.where(ok, eta, -np.inf)
    idx = np.unravel_index(np.argmax(eta_m), eta_m.shape)
    return dict(eta=eta_m[idx], E2=E2[idx], L2=L2[idx], E3=E3[idx], L3=L3[idx],
                dE=E3[idx] - E1)


def joint_search(a, alpha, m0=1.0, E_grid=None, L_grid=None, drmin=DRMIN_DEFAULT,
                  n_sample_r=12, n_split_grid=120, verbose=False):
    if E_grid is None:
        E_grid = np.linspace(0.6, 1.02, 14)
    if L_grid is None:
        L_grid = np.linspace(0.5, 4.5, 14)

    r_plus_0 = kv.r_plus(m0, a)
    global_best = None

    for E in E_grid:
        for L in L_grid:
            pr0 = pr_from_mass_shell(R0, a, m0, E, L, 'in')
            if pr0 is None:
                continue
            y0 = [0.0, R0, 0.0, -E, pr0, L]
            sol = integrate(y0, a, alpha, 250.0, r_plus_0)
            r_track, v_track, phi_track = sol.y[1], sol.y[0], sol.y[2]
            m_track = m_of_v(v_track, m0, alpha)
            rs_track = kv.r_s(m_track)
            rplus_track = np.array([kv.r_plus(mm, a) for mm in m_track])
            inside = (r_track < rs_track) & ((r_track - rplus_track) >= drmin)
            idxs = np.where(inside)[0]
            if len(idxs) == 0:
                continue
            sample_idxs = idxs[np.linspace(0, len(idxs) - 1, min(n_sample_r, len(idxs))).astype(int)]
            for si in sample_idxs:
                r_s_, v_s_, phi_s_, m_s_ = r_track[si], v_track[si], phi_track[si], m_track[si]
                cand = best_split_at_point(r_s_, m_s_, a, E, L, n_grid=n_split_grid)
                if cand is None:
                    continue
                if global_best is None or cand['eta'] > global_best['eta']:
                    cand.update(E1=E, L1=L, r_split=r_s_, v_split=v_s_, phi_split=phi_s_,
                                m_split=m_s_)
                    global_best = cand
            if verbose and global_best is not None:
                print(f"  E={E:.3f} L={L:.3f}: running best eta={global_best['eta']:.4f}")

    if global_best is None:
        return None

    # final escape verification on the winning candidate only
    ok = verify_escape(global_best['r_split'], global_best['v_split'],
                        global_best['phi_split'], global_best['E3'], global_best['L3'],
                        a, alpha, m0)
    global_best['escape_verified'] = ok
    return global_best


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = joint_search(0.9, 0.0, verbose=True)
    print(res)
    print("time:", time.time() - t0, "s")
