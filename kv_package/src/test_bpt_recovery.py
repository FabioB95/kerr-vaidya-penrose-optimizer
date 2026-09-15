"""test_bpt_recovery.py -- verifica se, imponendo la conservazione
COMPLETA del quadrimpulso al punto di split (non solo E e L, ma anche
p_r), l'ottimizzatore recupera il 20.7% classico all'estremalita'.

Se il rapporto NON tende a 1 con questo vincolo, il paper deve dire
onestamente "il nostro modello di split rilassa la conservazione di p_r
(consistente con Eq. 5, che il paper gia' dichiara solo per E e L)",
non "il bound BPT e' un caso piu' ristretto".
"""
import numpy as np
from scipy.optimize import fsolve
import sys
sys.path.insert(0, '.')
import geodesics as kv
from split_extraction import irreducible_mass


def mass_shell_norm(E, L, pr, ginv0, target_norm):
    """g^ab p_a p_b - target_norm, con p_v=-E, p_r=pr, p_phi=L."""
    gvv, gvr, gvp = ginv0[0, 0], ginv0[0, 1], ginv0[0, 2]
    grr, grp, gpp = ginv0[1, 1], ginv0[1, 2], ginv0[2, 2]
    return (gvv*E**2 - 2*gvr*E*pr - 2*gvp*E*L
            + grr*pr**2 + 2*grp*pr*L + gpp*L**2) - target_norm


def find_incoming_pr(E1, L1, r, ginv0):
    """pr1 per una particella massiva (norma -1) che cade (ramo ingoing)."""
    gvv, gvr, gvp = ginv0[0, 0], ginv0[0, 1], ginv0[0, 2]
    grr, grp, gpp = ginv0[1, 1], ginv0[1, 2], ginv0[2, 2]
    A = grr
    B = -2*gvr*E1 + 2*grp*L1
    C = gvv*E1**2 - 2*gvp*E1*L1 + gpp*L1**2 + 1  # norma -1
    disc = B**2 - 4*A*C
    if disc < 0:
        return None
    r1 = (-B - np.sqrt(disc)) / (2*A)
    r2 = (-B + np.sqrt(disc)) / (2*A)
    return min(r1, r2)  # ramo ingoing (pr<0 vicino all'orizzonte)


def full_conservation_equations(x, E1, L1, pr1):
    E3, L3, pr3 = x
    E2, L2, pr2 = E1 - E3, L1 - L3, pr1 - pr3
    eq_frag3 = mass_shell_norm(E3, L3, pr3, GINV, 0.0)   # fotone
    eq_frag2 = mass_shell_norm(E2, L2, pr2, GINV, 0.0)   # fotone
    return [eq_frag3, eq_frag2, 0.0]  # terza eq. placeholder, vedi sotto


def scan_full_conservation(r, m, a, E1, L1, pr1):
    """Scansiona pr3 e risolve (E3,L3) dalle 2 mass-shell equations via
    fsolve, con molti guess iniziali per catturare rami diversi."""
    global GINV
    GINV = np.array(kv.ginv_f(r, a, m, 0.0), dtype=float)
    Mirr_old = irreducible_mass(m, a)
    Ja = a * m

    best = None
    pr3_grid = np.linspace(-15, 15, 400)
    for pr3 in pr3_grid:
        for E3_guess in np.linspace(-2, 3, 6):
            for L3_guess in np.linspace(-6, 6, 6):
                def eqs(y):
                    E3, L3 = y
                    E2, L2, pr2 = E1-E3, L1-L3, pr1-pr3
                    return [mass_shell_norm(E3, L3, pr3, GINV, 0.0),
                            mass_shell_norm(E2, L2, pr2, GINV, 0.0)]
                sol, info, ier, msg = fsolve(eqs, [E3_guess, L3_guess],
                                              full_output=True, xtol=1e-12)
                if ier != 1:
                    continue
                E3, L3 = sol
                if max(abs(v) for v in eqs(sol)) > 1e-8:
                    continue
                E2, L2 = E1-E3, L1-L3
                if E2 >= 0:  # serve energia negativa per estrazione
                    continue
                M_new, J_new = m+E2, Ja+L2
                if M_new <= 0:
                    continue
                a_new = J_new / M_new
                if M_new**2 < a_new**2:
                    continue
                rp_new = M_new + np.sqrt(M_new**2 - a_new**2)
                Mirr_new = np.sqrt(rp_new**2 + a_new**2) / 2
                if Mirr_new < Mirr_old - 1e-9:
                    continue
                eta = (E3 - E1) / E1
                if best is None or eta > best[0]:
                    best = (eta, E3, L3, pr3)
    return best


if __name__ == "__main__":
    print(f"{'a/M':>10} {'eta_num':>12} {'eta_BPT':>12} {'rapporto':>10}")
    for a in [0.9, 0.99, 0.999, 0.9999]:
        m = 1.0
        rp = kv.r_plus(m, a)
        eta_theory = 0.5*(np.sqrt(2*m/rp) - 1)
        r = rp * 1.000001
        ginv0 = np.array(kv.ginv_f(r, a, m, 0.0), dtype=float)

        E1 = 1.0
        # L1 scansionato: per ogni L1 trova pr1 (se esiste) e prova lo split
        best_overall = None
        for L1 in np.linspace(-2, 6, 30):
            pr1 = find_incoming_pr(E1, L1, r, ginv0)
            if pr1 is None:
                continue
            res = scan_full_conservation(r, m, a, E1, L1, pr1)
            if res and (best_overall is None or res[0] > best_overall[0]):
                best_overall = res
        if best_overall is None:
            print(f"{a:>10} {'nessuno split valido trovato':>30}")
            continue
        eta_num = best_overall[0]
        print(f"{a:>10.4f} {eta_num*100:>11.4f}% {eta_theory*100:>11.4f}% "
              f"{eta_num/eta_theory:>10.3f}")