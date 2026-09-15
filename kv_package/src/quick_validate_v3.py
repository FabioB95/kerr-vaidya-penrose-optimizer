"""quick_validate_v3.py -- verifica il modulo vettorizzato contro BPT,
usando la stessa configurazione (E1=1, particella radiale) di
test_bpt_recovery.py ma con la funzione veloce. Deve dare essenzialmente
lo stesso rapporto=1.000 in una frazione del tempo."""
import numpy as np
import sys
sys.path.insert(0, '.')
import geodesics as kv
from full_split import best_split_full_conservation


def find_incoming_pr(E1, L1, r, ginv0):
    gvv, gvr, gvp = ginv0[0, 0], ginv0[0, 1], ginv0[0, 2]
    grr, grp, gpp = ginv0[1, 1], ginv0[1, 2], ginv0[2, 2]
    A = grr
    B = -2 * gvr * E1 + 2 * grp * L1
    C = gvv * E1**2 - 2 * gvp * E1 * L1 + gpp * L1**2 + 1  # norma -1 (massiva)
    disc = B**2 - 4 * A * C
    if disc < 0:
        return None
    r1 = (-B - np.sqrt(disc)) / (2 * A)
    r2 = (-B + np.sqrt(disc)) / (2 * A)
    return min(r1, r2)


print(f"{'a/M':>10} {'eta_num':>12} {'eta_BPT':>12} {'rapporto':>10}")
for a in [0.9, 0.99, 0.999, 0.9999]:
    m = 1.0
    rp = kv.r_plus(m, a)
    eta_theory = 0.5 * (np.sqrt(2 * m / rp) - 1)
    r = rp * 1.000001
    ginv0 = np.array(kv.ginv_f(r, a, m, 0.0), dtype=float)

    E1 = 1.0
    best_overall = None
    for L1 in np.linspace(-2, 6, 60):
        pr1 = find_incoming_pr(E1, L1, r, ginv0)
        if pr1 is None:
            continue
        res = best_split_full_conservation(r, m, a, E1, L1, pr1)
        if res and (best_overall is None or res['eta'] > best_overall['eta']):
            best_overall = res
    if best_overall is None:
        print(f"{a:>10} {'nessuno split trovato':>30}")
        continue
    eta_num = best_overall['eta']
    print(f"{a:>10.4f} {eta_num*100:>11.4f}% {eta_theory*100:>11.4f}% "
          f"{eta_num/eta_theory:>10.3f}")