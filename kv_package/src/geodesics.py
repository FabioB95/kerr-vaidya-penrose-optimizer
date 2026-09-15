"""
Corrected equatorial Kerr-Vaidya geodesic integrator.
Hamiltonian form:  dx^mu/dtau = g^{mu nu} p_nu
                    dp_mu/dtau = -1/2 (d_mu g^{ab}) p_a p_b
avoids building Christoffels explicitly (less error-prone), uses the
CORRECTED metric with g_{r phi} = -a (constant), verified against
direct BL -> advanced-coordinate transform in derive_metric.py.
"""
import sympy as sp
import numpy as np

r, phi, a, m, mdot = sp.symbols('r phi a m mdot', real=True)
Sigma = r**2  # equatorial reduction of r^2 + a^2*cos^2(theta) at theta=pi/2
Delta = r**2 - 2*m*r + a**2

g = sp.zeros(3, 3)  # order (v, r, phi)
g[0, 0] = -(1 - 2*m*r/Sigma)
g[0, 1] = g[1, 0] = 1
g[0, 2] = g[2, 0] = -2*a*m*r/Sigma
g[1, 1] = 0
g[1, 2] = g[2, 1] = -a
g[2, 2] = r**2 + a**2 + 2*m*r*a**2/Sigma

ginv = sp.simplify(g.inv())

# partial derivatives of g^{munu}: d/dv = mdot * d/dm ;  d/dr direct ;  d/dphi = 0
dginv_dm = ginv.applyfunc(lambda e: sp.diff(e, m))
dginv_dr = ginv.applyfunc(lambda e: sp.diff(e, r))

vars_ = (r, a, m, mdot)
ginv_f = sp.lambdify(vars_, ginv, 'numpy')
dginv_dv_f = sp.lambdify(vars_, dginv_dm * mdot, 'numpy')  # chain rule, elementwise scale below
dginv_dm_f = sp.lambdify(vars_, dginv_dm, 'numpy')
dginv_dr_f = sp.lambdify(vars_, dginv_dr, 'numpy')

r_plus = lambda m_, a_: m_ + np.sqrt(max(m_**2 - a_**2, 0.0))
r_s = lambda m_: 2*m_


def rhs(tau, y, a_, alpha):
    """y = [v, r, phi, pv, pr, pphi]. mdot = -alpha (mass loss)."""
    v, r_, phi_, pv, pr, pph = y
    m_ = m0_global - alpha * v
    mdot_ = -alpha
    ginv_num = np.array(ginv_f(r_, a_, m_, mdot_), dtype=float)
    dg_dv = np.array(dginv_dm_f(r_, a_, m_, mdot_), dtype=float) * mdot_
    dg_dr = np.array(dginv_dr_f(r_, a_, m_, mdot_), dtype=float)

    p = np.array([pv, pr, pph])
    dxdtau = ginv_num @ p

    dpv = -0.5 * (p @ dg_dv @ p)
    dpr = -0.5 * (p @ dg_dr @ p)
    dpph = 0.0  # phi is a Killing vector: p_phi = L conserved exactly

    return [dxdtau[0], dxdtau[1], dxdtau[2], dpv, dpr, dpph]


def mass_shell_residual(y, a_, m_):
    v, r_, phi_, pv, pr, pph = y
    ginv_num = np.array(ginv_f(r_, a_, m_, 0.0), dtype=float)
    p = np.array([pv, pr, pph])
    return p @ ginv_num @ p + 1.0  # should be 0 (norm -1)


m0_global = 1.0  # set by caller before integrating; module-level for simplicity

if __name__ == "__main__":
    # --- Verification 1: static Kerr, a/M=0.9, E1=0.95 particle from r0=6M,
    #     check p_v = -E is conserved (energy conservation in static limit)
    from scipy.integrate import solve_ivp

    a_ = 0.9
    m0_global = 1.0
    E1 = 0.95
    # radial momentum from mass-shell for purely radial infall with L=0 at r0=6
    L1 = 2.0
    r0 = 6.0
    ginv0 = np.array(ginv_f(r0, a_, 1.0, 0.0), dtype=float)
    pv0, pph0 = -E1, L1
    # solve quadratic for pr from mass shell: ginv[1,1]*pr^2 + 2*ginv[1,0]*pv*pr + 2*ginv[1,2]*pph*pr
    #    + ginv[0,0]pv^2+2ginv[0,2]pv*pph+ginv[2,2]pph^2 + 1 = 0
    A = ginv0[1, 1]
    B = 2*ginv0[1, 0]*pv0 + 2*ginv0[1, 2]*pph0
    C = ginv0[0, 0]*pv0**2 + 2*ginv0[0, 2]*pv0*pph0 + ginv0[2, 2]*pph0**2 + 1
    if abs(A) < 1e-14:
        pr0 = -C / B
    else:
        disc = B**2 - 4*A*C
        pr0 = (-B - np.sqrt(disc)) / (2*A)  # ingoing root

    y0 = [0.0, r0, 0.0, pv0, pr0, pph0]
    sol = solve_ivp(rhs, [0, 50], y0, args=(a_, 0.0), max_step=0.01,
                     rtol=1e-10, atol=1e-12, dense_output=True)
    pv_series = sol.y[3]
    print("static Kerr check: p_v (=-E) drift over integration:",
          pv_series.max() - pv_series.min(), " (should be ~0)")
    print("mass-shell residual at start:", mass_shell_residual(y0, a_, 1.0))
    print("mass-shell residual at end:", mass_shell_residual(sol.y[:, -1], a_, 1.0))
    print("r range reached:", sol.y[1].min(), "-", sol.y[1].max())
    print("r_plus =", r_plus(1.0, a_), " r_s =", r_s(1.0))
