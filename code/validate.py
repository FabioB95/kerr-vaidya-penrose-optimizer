"""Validation suite for kvcore (results are written to data/validation.txt)."""
import numpy as np
import kvcore as K

out = []
def log(s):
    print(s); out.append(s)

def par(law, alpha, dm, spin):
    return np.array([float(law), alpha, dm, spin, 1.0, 0.0])

# 1. finite-difference check of Hamilton's equations ------------------------
rng = np.random.default_rng(1)
maxerr = 0.0
for law in (0, 1):
    P = par(law, 0.01, 0.05, 0.9 if law == 0 else 0.9)
    for _ in range(200):
        y = np.array([rng.uniform(0.1, 4.9), rng.uniform(1.5, 8), 0.3,
                      rng.uniform(-1.5, -0.5), rng.uniform(-2, 2), rng.uniform(-4, 4)])
        f = np.empty(6); K.rhs(y, P, f)
        num = np.empty(6)
        for i in range(6):
            e = 1e-6
            yp = y.copy(); ym = y.copy(); yp[i] += e; ym[i] -= e
            d = (K.hamiltonian(yp, P) - K.hamiltonian(ym, P)) / (2 * e)
            num[i] = d
        # dx/dtau = dH/dp ; dp/dtau = -dH/dx
        ref = np.array([num[3], num[4], num[5], -num[0], -num[1], -num[2]])
        maxerr = max(maxerr, np.max(np.abs(ref - f) / (1 + np.abs(ref))))
        # analytic identity
        ide = K.energy_rate_identity(y, P)
        maxerr = max(maxerr, abs(ide - (-f[3])) / (1 + abs(ide)))
log(f"[1] Hamilton eqs vs finite-difference dH and identity: max rel err = {maxerr:.2e}")

# 2. static Kerr: E, L, H conserved -----------------------------------------
P = par(0, 0.0, 0.0, 0.9)
pr = K.pr_from_shell(10.0, 1.0, 0.9, -1.0, 2.5, 1.0, 'in')
y0 = np.array([0.0, 10.0, 0.0, -1.0, pr, 2.5])
H0 = K.hamiltonian(y0, P)
y, tau, st, rmin, tr, nt = K.integrate(y0, P, 2000.0, 300.0, 0, 0.0, record=200000)
dE = np.max(np.abs(tr[:nt, 3] + 1.0))
dH = max(abs(K.hamiltonian(tr[i, :6], P) - H0) for i in range(0, nt, 50))
log(f"[2] static Kerr a=0.9 flyby: status={st}, rmin={rmin:.3f}, max|dE|={dE:.1e}, max|dH|={dH:.1e}")

# 3. dynamic: H conserved, E(tau) vs integral of identity --------------------
for law, spin in ((0, 0.9), (1, 0.9)):
    P = par(law, 2e-3, 0.05, spin)
    y, tau, st, rmin, tr, nt = K.integrate(y0, P, 4000.0, 300.0, 0, 0.0, record=400000)
    H = np.array([K.hamiltonian(tr[i, :6], P) for i in range(nt)])
    rate = np.array([K.energy_rate_identity(tr[i, :6], P) for i in range(nt)])
    Eint = -y0[3] + np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(tr[:nt, 6]))])
    err = np.max(np.abs(Eint - (-tr[:nt, 3])))
    log(f"[3] law {'AB'[law]} alpha=2e-3: status={st}, E_final={-y[3]:.6f} (E0=1), "
        f"max|dH|={np.max(np.abs(H-H0)):.1e}, |E - (E0+int dE/dtau)|_max={err:.1e}")

# 4. decay kinematics and BPT bound -----------------------------------------
for a in (0.9, 0.99, 0.999, 0.9999):
    m = 1.0
    rp = K.r_plus(m, a)
    r = rp * (1 + 1e-7)
    P = par(0, 0.0, 0.0, a)
    # incoming particle with local E1=1, L1=0 (ingoing)
    pr1 = K.pr_from_shell(r, m, a, -1.0, 0.0, 1.0, 'in')
    y = np.array([0.0, r, 0.0, -1.0, pr1, 0.0])
    p2, p3 = K.decay(y, P, 1.0, 0.0)
    Gi = K.ginv_eq(r, m, a)
    cons = np.max(np.abs(p2 + p3 - y[3:]))
    n2 = p2 @ Gi @ p2; n3 = p3 @ Gi @ p3
    eta = -(-p2[0]) / 1.0
    bpt = 0.5 * (np.sqrt(2 * m / rp) - 1)
    log(f"[4] a={a}: eta_loc={eta:.6f}  BPT={bpt:.6f}  |p1-p2-p3|={cons:.1e}  "
        f"p2^2={n2:.1e} p3^2={n3:.1e}")

with open('../data/validation.txt', 'w') as f:
    f.write("\n".join(out) + "\n")
