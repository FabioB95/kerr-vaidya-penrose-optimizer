import numpy as np
from scipy.integrate import solve_ivp
import geodesics as kv

R0 = 6.0
R_ESC = 20.0
R_ESC_VERIFY_TAU = 500.0
DRMIN_DEFAULT = 0.3


def m_of_v(v, m0, alpha):
    return m0 - alpha * v


def pr_from_mass_shell(r, a, m, E, L, branch):
    ginv0 = np.array(kv.ginv_f(r, a, m, 0.0), dtype=float)
    pv0, pph0 = -E, L
    A = ginv0[1, 1]
    B = 2*ginv0[1, 0]*pv0 + 2*ginv0[1, 2]*pph0
    C = ginv0[0, 0]*pv0**2 + 2*ginv0[0, 2]*pv0*pph0 + ginv0[2, 2]*pph0**2 + 1
    disc = B**2 - 4*A*C
    if disc < 0:
        return None
    root1 = (-B - np.sqrt(disc)) / (2*A)
    root2 = (-B + np.sqrt(disc)) / (2*A)
    lo, hi = min(root1, root2), max(root1, root2)
    return lo if branch == 'in' else hi


def irreducible_mass(M, a_spin):
    if M**2 < a_spin**2:
        return None
    rp = M + np.sqrt(M**2 - a_spin**2)
    return np.sqrt(rp**2 + a_spin**2) / 2.0


def area_theorem_ok(M, J, E2, L2):
    Mirr_old = irreducible_mass(M, J / M)
    M_new = M + E2
    if M_new <= abs(J + L2) or M_new <= 0:
        return False
    J_new = J + L2
    a_new = J_new / M_new
    Mirr_new = irreducible_mass(M_new, a_new)
    if Mirr_new is None:
        return False
    return Mirr_new >= Mirr_old - 1e-9


def integrate(y0, a, alpha, tau_max, r_plus_stop):
    def horizon_event(tau, y, a_, alpha_):
        return y[1] - (r_plus_stop + 0.01)
    horizon_event.terminal = True
    horizon_event.direction = -1

    def escape_event(tau, y, a_, alpha_):
        return y[1] - R_ESC
    escape_event.terminal = True
    escape_event.direction = 1

    sol = solve_ivp(kv.rhs, [0, tau_max], y0, args=(a, alpha), max_step=0.05,
                     rtol=1e-8, atol=1e-10, events=[horizon_event, escape_event])
    return sol


def verify_escape(r, v, phi, E, L, a, alpha, m0):
    m_here = m_of_v(v, m0, alpha)
    pr = pr_from_mass_shell(r, a, m_here, E, L, 'out')
    if pr is None:
        return False
    y0 = [v, r, phi, -E, pr, L]
    r_plus_now = kv.r_plus(m_here, a)
    sol = integrate(y0, a, alpha, R_ESC_VERIFY_TAU, r_plus_now)
    return sol.y[1, -1] >= R_ESC - 1e-6


def skimming_search(a, alpha, m0, drmin=DRMIN_DEFAULT, E_grid=None, L_grid=None,
                     tau_max=250.0):
    if E_grid is None:
        E_grid = np.linspace(0.6, 1.02, 18)
    if L_grid is None:
        L_grid = np.linspace(0.5, 4.5, 18)

    best = None
    r_plus_0 = kv.r_plus(m0, a)
    for E in E_grid:
        for L in L_grid:
            pr0 = pr_from_mass_shell(R0, a, m0, E, L, 'in')
            if pr0 is None:
                continue
            y0 = [0.0, R0, 0.0, -E, pr0, L]
            sol = integrate(y0, a, alpha, tau_max, r_plus_0)
            r_track = sol.y[1]
            v_track = sol.y[0]
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
            rmin = r_track[idx]
            score = (rs_track[idx] - rmin) * np.sum(inside)
            if best is None or score > best['score']:
                best = dict(score=score, E=E, L=L, r_split=r_track[idx],
                            v_split=v_track[idx], phi_split=sol.y[2][idx],
                            m_split=m_track[idx])
    return best


def split_optimize_fast(skim, a, alpha, m0, E1, L1, n_grid=60, n_verify=8):
    """Vectorized filter over (E3,L3) grid via numpy, THEN escape-verify only
    the top n_verify candidates by eta (avoids one ODE call per grid point)."""
    r_split, v_split, phi_split, m_split = (skim['r_split'], skim['v_split'],
                                             skim['phi_split'], skim['m_split'])
    r_plus_now = kv.r_plus(m_split, a)
    pr3_max = 15.0 if (r_split - r_plus_now < 0.25) else 10.0

    ginv0 = np.array(kv.ginv_f(r_split, a, m_split, 0.0), dtype=float)
    E3g = np.linspace(-1.0, 2.5, n_grid)
    L3g = np.linspace(-4.0, 6.0, n_grid)
    E3, L3 = np.meshgrid(E3g, L3g, indexing='ij')
    E2 = E1 - E3
    L2 = L1 - L3

    def veff_grid(E, L):
        pv0, pph0 = -E, L
        A = ginv0[1, 1]
        B = 2*ginv0[1, 0]*pv0 + 2*ginv0[1, 2]*pph0
        C = ginv0[0, 0]*pv0**2 + 2*ginv0[0, 2]*pv0*pph0 + ginv0[2, 2]*pph0**2 + 1
        return -(B**2 - 4*A*C)

    ok_local = (veff_grid(E2, L2) <= 1e-9) & (veff_grid(E3, L3) <= 1e-9)

    J_here = a * m_split
    Mirr_old = irreducible_mass(m_split, a)
    M_new = m_split + E2
    J_new = J_here + L2
    with np.errstate(invalid='ignore'):
        a_new = J_new / M_new
        rp_new = M_new + np.sqrt(np.maximum(M_new**2 - a_new**2, np.nan))
        Mirr_new = np.sqrt(rp_new**2 + a_new**2) / 2.0
    valid_mass = (M_new > 0) & (M_new**2 >= a_new**2) & (M_new > np.abs(J_new))
    ok_area = valid_mass & (Mirr_new >= Mirr_old - 1e-9)

    ok = ok_local & ok_area
    eta = (E3 - E1) / E1
    eta_masked = np.where(ok, eta, -np.inf)

    flat_idx = np.argsort(eta_masked.ravel())[::-1]
    best = None
    tried = 0
    for fi in flat_idx:
        if tried >= n_verify:
            break
        i, j = np.unravel_index(fi, eta_masked.shape)
        if not ok[i, j]:
            break  # sorted descending; once we hit invalid, rest are worse/invalid
        tried += 1
        e3v, l3v = E3[i, j], L3[i, j]
        if verify_escape(r_split, v_split, phi_split, e3v, l3v, a, alpha, m0):
            best = dict(eta=eta[i, j], E2=E2[i, j], L2=L2[i, j], E3=e3v, L3=l3v,
                        dE=e3v - E1, r_split=r_split, v_split=v_split)
            break
    return best


def run_case(a, alpha, m0=1.0, E_grid=None, L_grid=None, drmin=DRMIN_DEFAULT,
             n_split_grid=60, n_verify=8):
    kv.m0_global = m0
    skim = skimming_search(a, alpha, m0, drmin=drmin, E_grid=E_grid, L_grid=L_grid)
    if skim is None:
        return None
    result = split_optimize_fast(skim, a, alpha, m0, skim['E'], skim['L'],
                                  n_grid=n_split_grid, n_verify=n_verify)
    if result is None:
        return None
    result['E1'] = skim['E']
    result['L1'] = skim['L']
    return result


if __name__ == "__main__":
    import time
    t0 = time.time()
    a = 0.9
    res = run_case(a, 0.0)
    print(f"static a/M={a}: {res}")
    print("time:", time.time() - t0, "s")
