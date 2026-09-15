import numpy as np
from scipy.optimize import brentq
import geodesics as kv
from optimizer2 import irreducible_mass


def disc_of_E(E, L, ginv0):
    pv0, pph0 = -E, L
    A = ginv0[1, 1]
    B = 2*ginv0[1, 0]*pv0 + 2*ginv0[1, 2]*pph0
    C = ginv0[0, 0]*pv0**2 + 2*ginv0[0, 2]*pv0*pph0 + ginv0[2, 2]*pph0**2 + 1
    return B**2 - 4*A*C


def E_lower_bound(L, ginv0, E_scan=None):
    """For fixed L, find the lower edge of the E-interval where disc(E,L)>=0
    (i.e. the most negative E for which the fragment's momentum is still
    real) by bracketing the sign change and using brentq."""
    if E_scan is None:
        E_scan = np.linspace(-3.0, 3.0, 400)
    vals = np.array([disc_of_E(E, L, ginv0) for E in E_scan])
    sign = np.sign(vals)
    roots = []
    for i in range(len(E_scan) - 1):
        if sign[i] == 0:
            roots.append(E_scan[i])
        elif sign[i] != sign[i+1] and sign[i] != 0 and sign[i+1] != 0:
            r = brentq(disc_of_E, E_scan[i], E_scan[i+1], args=(L, ginv0))
            roots.append(r)
    if not roots:
        return None
    return min(roots)  # most negative real-boundary E


def boundary_search(r, m_here, a, E1, L1, L_grid=None):
    """Scan L2 along the physical boundary; for each L2, take the most
    negative allowed E2(L2), check the area theorem, track best efficiency."""
    ginv0 = np.array(kv.ginv_f(r, a, m_here, 0.0), dtype=float)
    if L_grid is None:
        L_grid = np.linspace(-6.0, 6.0, 4000)

    Mirr_old = irreducible_mass(m_here, a)
    J_here = a * m_here
    best = None
    for L2 in L_grid:
        E2 = E_lower_bound(L2, ginv0)
        if E2 is None:
            continue
        M_new = m_here + E2
        J_new = J_here + L2
        if M_new <= 0 or M_new**2 < (J_new/M_new if M_new != 0 else 1e9)**2:
            pass
        a_new = J_new / M_new if M_new != 0 else None
        if a_new is None or M_new**2 < a_new**2 or M_new <= 0:
            continue
        rp_new = M_new + np.sqrt(M_new**2 - a_new**2)
        Mirr_new = np.sqrt(rp_new**2 + a_new**2) / 2.0
        if Mirr_new < Mirr_old - 1e-10:
            continue
        E3, L3 = E1 - E2, L1 - L2
        # fragment 3 must also be locally real
        if disc_of_E(E3, L3, ginv0) < -1e-9:
            continue
        eta = (E3 - E1) / E1
        if best is None or eta > best['eta']:
            best = dict(eta=eta, E2=E2, L2=L2, E3=E3, L3=L3, dE=E3 - E1)
    return best


if __name__ == "__main__":
    a = 0.9
    m = 1.0
    rp = kv.r_plus(m, a)
    E1, L1 = 0.95, 2.0
    for margin in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
        r = rp + margin
        res = boundary_search(r, m, a, E1, L1)
        if res:
            print(f"margin={margin:.4f}  eta={res['eta']*100:.3f}%  dE={res['dE']:.5f}  "
                  f"E2={res['E2']:.5f} L2={res['L2']:.4f}")
        else:
            print(f"margin={margin:.4f}  no valid split found")
