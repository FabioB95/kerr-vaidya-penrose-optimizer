"""pipeline_v3.py -- diagnostica rigorosa"""
import numpy as np
import geodesics as kv
from pipeline import pr_from_mass_shell, integrate, m_of_v, R0
from full_split import all_valid_escaping_splits, all_valid_splits, drdtau_sign


def verify_escape_photon(r, v, phi, E, L, pr, a, alpha, m0,
                         r_esc=20.0, tau_max=800.0):
    y0 = [v, r, phi, -E, pr, L]
    r_plus_now = kv.r_plus(m_of_v(v, m0, alpha), a)
    sol = integrate(y0, a, alpha, tau_max, r_plus_now, horizon_margin=1e-5)
    return sol.y[1, -1] >= r_esc - 1e-5


def run_case_v3(a=0.9, alpha=0.0, m0=1.0, drmin=0.05,
                 tau_max=400.0, verbose=True):

    E_grid = np.linspace(0.7, 1.05, 12)
    L_grid = np.linspace(-2.0, 5.0, 16)

    kv.m0_global = m0
    r_plus_0 = kv.r_plus(m0, a)

    total_traj = 0
    traj_with_safe = 0
    points_tested = 0
    points_with_local = 0
    points_with_escaping_local = 0
    verified_ok = 0
    best = None

    for E in E_grid:
        for L in L_grid:
            total_traj += 1
            pr0 = pr_from_mass_shell(R0, a, m0, E, L, 'in')
            if pr0 is None:
                continue

            y0 = [0.0, R0, 0.0, -E, pr0, L]
            sol = integrate(y0, a, alpha, tau_max, r_plus_0, horizon_margin=1e-5)

            r_track   = sol.y[1]
            v_track   = sol.y[0]
            phi_track = sol.y[2]
            pv_track  = sol.y[3]
            pr_track  = sol.y[4]
            pph_track = sol.y[5]
            m_track   = m_of_v(v_track, m0, alpha)

            rs_track    = kv.r_s(m_track)
            rplus_track = np.array([kv.r_plus(mm, a) for mm in m_track])
            inside      = r_track < rs_track
            if not np.any(inside):
                continue

            margin = r_track - rplus_track
            safe   = inside & (margin >= drmin)
            idxs   = np.where(safe)[0]
            if len(idxs) == 0:
                continue

            traj_with_safe += 1
            deepest = idxs[np.argsort(r_track[idxs])[:6]]

            for si in deepest:
                points_tested += 1
                r_s_ = r_track[si]
                v_s_ = v_track[si]
                phi_s_ = phi_track[si]
                m_s_ = m_track[si]
                E1   = -pv_track[si]
                L1   =  pph_track[si]
                pr1  =  pr_track[si]
                marg = r_s_ - kv.r_plus(m_s_, a)

                cands_all = all_valid_splits(r_s_, m_s_, a, E1, L1, pr1,
                                             n_L3=300, n_E3=250)
                if cands_all:
                    points_with_local += 1

                cands_esc = all_valid_escaping_splits(r_s_, m_s_, a, E1, L1, pr1,
                                                      n_L3=300, n_E3=250)
                if not cands_esc:
                    continue

                points_with_escaping_local += 1
                if verbose:
                    print(f"  LOCAL ESCAPING: r={r_s_:.4f}  E1={E1:.3f}  L1={L1:.3f}  "
                          f"margin={marg:.4f}  n={len(cands_esc)}  "
                          f"best_eta={cands_esc[0]['eta']*100:.2f}%")

                for cand in cands_esc[:3]:
                    ok = verify_escape_photon(r_s_, v_s_, phi_s_,
                                              cand['E3'], cand['L3'], cand['pr3'],
                                              a, alpha, m0)
                    if ok:
                        verified_ok += 1
                        if best is None or cand['eta'] > best['eta']:
                            best = dict(
                                eta=cand['eta'], dE=cand['E3']-E1,
                                E1=E1, L1=L1, pr1=pr1,
                                E2=cand['E2'], L2=cand['L2'], pr2=cand['pr2'],
                                E3=cand['E3'], L3=cand['L3'], pr3=cand['pr3'],
                                r_split=r_s_, v_split=v_s_, phi_split=phi_s_,
                                m_split=m_s_, margin=marg
                            )
                            print(f"    >>> VERIFIED BEST eta={best['eta']*100:.2f}%")
                        break

    print("\n========== RIEPILOGO DIAGNOSTICO ==========")
    print(f"Traiettorie totali:                     {total_traj}")
    print(f"Traiettorie con punti safe:             {traj_with_safe}")
    print(f"Punti testati:                          {points_tested}")
    print(f"Punti con almeno 1 candidato locale:    {points_with_local}")
    print(f"Punti con candidato localmente uscente: {points_with_escaping_local}")
    print(f"Candidati che hanno superato verify:    {verified_ok}")
    print("==========================================")
    return best


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = run_case_v3(verbose=True)
    print("\nFinal:", res)
    print("time:", time.time() - t0, "s")