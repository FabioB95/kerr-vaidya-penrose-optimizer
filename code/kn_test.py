"""Independent numerical test of Corollary 2 (Kerr--Newman--Vaidya).
The equatorial Hamiltonian is built symbolically from the full metric with
m(v) and Q(v) (independent of kvcore), integrated with scipy DOP853, and the
energy E(tau) = -p_v is compared with E0 + integral of
dE/dtau = -(mdot r - Q Qdot) p_r^2 / r^2   (Sigma = r^2 on the equator)."""
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp
v, r, pv, pr, pp = sp.symbols('v r p_v p_r p_phi', real=True)
a = sp.Rational(9, 10)
al, be = sp.symbols('alpha beta', real=True)
m = 1 - al * v
Q = sp.Rational(3, 10) - be * v
W = 2 * m * r - Q**2
D = r**2 - 2 * m * r + a**2 + Q**2
g = sp.Matrix([[-(1 - W / r**2), 1, -a * W / r**2],
               [1, 0, -a],
               [-a * W / r**2, -a, ((r**2 + a**2)**2 - D * a**2) / r**2]])
gi = sp.simplify(g.inv())
p = sp.Matrix([pv, pr, pp])
H = sp.simplify((p.T * gi * p)[0] / 2)
X = (v, r, pv, pr, pp)
rhs = [sp.diff(H, pv), sp.diff(H, pr), sp.diff(H, pp), -sp.diff(H, v), -sp.diff(H, r)]
f = sp.lambdify(X + (al, be), rhs, 'numpy')
Hf = sp.lambdify(X + (al, be), H, 'numpy')
rate = sp.lambdify(X + (al, be), -(sp.diff(m, v) * r - Q * sp.diff(Q, v)) * pr**2 / r**2, 'numpy')
out = []
for alpha, beta in ((2e-3, 0.0), (0.0, 2e-3), (2e-3, 2e-3), (2e-3, -2e-3)):
    # incoming massive particle at r=10, E=1, L=3: solve mass shell for p_r (ingoing)
    sol_pr = sp.solve(sp.Eq(H.subs({v: 0, r: 10, pv: -1, pp: 3, al: alpha, be: beta}), -sp.Rational(1, 2)), pr)
    pr0 = min(float(s_) for s_ in sol_pr)
    y0 = [0.0, 10.0, -1.0, pr0, 3.0]
    F = lambda t, y: [0.0 if False else 0.0] or f(*y[:2], y[2], y[3], y[4], alpha, beta)
    def F(t, y):
        dv, dr, dph, dpv, dpr = f(y[0], y[1], y[2], y[3], y[4], alpha, beta)
        return [dv, dr, dpv, dpr, 0.0]
    ev = lambda t, y: y[1] - 40.0
    ev.terminal = True
    s = solve_ivp(F, [0, 300], y0, method='DOP853', rtol=1e-12, atol=1e-14, events=ev, dense_output=True)
    tt = np.linspace(0, s.t[-1], 20001)
    Y = s.sol(tt)
    R = rate(Y[0], Y[1], Y[2], Y[3], 3.0, alpha, beta)
    Eint = 1.0 + np.concatenate([[0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(tt))])
    err = np.max(np.abs(Eint - (-Y[2])))
    dH = np.max(np.abs(Hf(Y[0], Y[1], Y[2], Y[3], 3.0, alpha, beta) + 0.5))
    msg = (f"mdot={-alpha:+.0e}, Qdot={-beta:+.0e}: rmin={Y[1].min():.3f}, "
           f"E_final={-Y[2][-1]:.6f}, |E - (E0+int rate)|_max={err:.1e}, |dH|_max={dH:.1e}")
    out.append(msg); print(msg)
open('../data/kn_test.txt', 'w').write("\n".join(out) + "\n")
