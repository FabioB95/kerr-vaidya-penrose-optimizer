import numpy as np
from scipy.integrate import solve_ivp
import geodesics as kv
from split_extraction import max_extraction, irreducible_mass

R0 = 6.0
DRMIN_DEFAULT = 0.3
R_ESC = 20.0
R_ESC_VERIFY_TAU = 500.0


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


def integrate(y0, a, alpha, tau_max, r_plus_stop, horizon_margin=0.005):
    def horizon_event(tau, y, a_, alpha_):
        return y[1] - (r_plus_stop + horizon_margin)
    horizon_event.terminal = True
    horizon_event.direction = -1

    def escape_event(tau, y, a_, alpha_):
        return y[1] - R_ESC
    escape_event.terminal = True
    escape_event.direction = 1

    sol = solve_ivp(kv.rhs, [0, tau_max], y0, args=(a, alpha), max_step=0.05,
                     rtol=1e-9, atol=1e-11, events=[horizon_event, escape_event])
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
        E_grid = np.linspace(0.5, 1.05, 20)
    if L_grid is None:
        L_grid = np.linspace(-1.0, 4.5, 24)

    best = None
    r_plus_0 = kv.r_plus(m0, a)
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
            rmin = r_track[idx]
            score = (rs_track[idx] - rmin) * np.sum(inside)
            if best is None or score > best['score']:
                best = dict(score=score, E=E, L=L, r_split=r_track[idx],
                            v_split=v_track[idx], phi_split=phi_track[idx],
                            m_split=m_track[idx])
    return best


def run_case(a, alpha, m0=1.0, drmin=DRMIN_DEFAULT, E_grid=None, L_grid=None,
             verify=True):
    kv.m0_global = m0
    skim = skimming_search(a, alpha, m0, drmin=drmin, E_grid=E_grid, L_grid=L_grid)
    if skim is None:
        return None
    ext = max_extraction(skim['r_split'], skim['m_split'], a)
    if ext is None:
        return None
    E1 = skim['E']
    result = dict(dE=ext['dE'], eta=ext['dE'] / E1, E1=E1, L1=skim['L'],
                  E2=ext['E2'], L2=ext['L2'], ell2=ext['ell2'],
                  r_split=skim['r_split'], v_split=skim['v_split'],
                  phi_split=skim['phi_split'], m_split=skim['m_split'])
    if verify:
        E3 = E1 - ext['E2']
        L3 = skim['L'] - ext['L2']
        result['escape_verified'] = verify_escape(
            skim['r_split'], skim['v_split'], skim['phi_split'], E3, L3, a, alpha, m0)
        result['E3'], result['L3'] = E3, L3
    return result


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = run_case(0.9, 0.0)
    print("static a/M=0.9:", res)
    print("time:", time.time() - t0, "s")