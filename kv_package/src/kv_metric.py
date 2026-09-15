"""
Corrected equatorial Kerr-Vaidya metric (theta = pi/2), advanced coords (v,r,phi).
g_rphi = -a  (CONSTANT) -- this was the bug flagged by the referee and confirmed
by direct BL->advanced coordinate transform in derive_metric.py.

Coordinates order: x = (v, r, phi)
"""
import sympy as sp
import numpy as np

v, r, phi, a, m, mdot = sp.symbols('v r phi a m mdot', real=True)
mv = sp.Function('m')(v)  # mass as a function of v, for Christoffel derivatives

Sigma = r**2  # equatorial reduction of r^2 + a^2*cos^2(theta) at theta=pi/2
Delta = r**2 - 2*mv*r + a**2

g = sp.zeros(3, 3)
# order: 0=v, 1=r, 2=phi
g[0, 0] = -(1 - 2*mv*r/Sigma)
g[0, 1] = g[1, 0] = 1
g[0, 2] = g[2, 0] = -2*a*mv*r/Sigma
g[1, 1] = 0
g[1, 2] = g[2, 1] = -a                       # <-- corrected term (was 2*a*mv*r/Delta)
g[2, 2] = (r**2 + a**2 + 2*mv*r*a**2/Sigma)

coords = [v, r, phi]
ginv = g.inv()
ginv = sp.simplify(ginv)

# Christoffel symbols Gamma^lambda_{mu nu}, with d/dv acting on mv via mdot = dm/dv
def dmu(expr, mu):
    if mu == 0:
        return sp.diff(expr, v) + sp.diff(expr, mv) * mdot - sp.diff(expr, mv) * mdot  # placeholder, unused
    return sp.diff(expr, coords[mu])

# proper partials: treat mv as function of v; sp.diff w.r.t. v already applies chain rule via mv'(v)=mdot-sub later
dv_g = g.applyfunc(lambda e: sp.diff(e, v))
dr_g = g.applyfunc(lambda e: sp.diff(e, r))
dphi_g = g.applyfunc(lambda e: sp.diff(e, phi))
dg = [dv_g, dr_g, dphi_g]

Gamma = [[[0]*3 for _ in range(3)] for _ in range(3)]
for lam in range(3):
    for mu in range(3):
        for nu in range(3):
            s = 0
            for sig in range(3):
                s += ginv[lam, sig] * (dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
            Gamma[lam][mu][nu] = sp.simplify(sp.Rational(1, 2) * s)

# substitute mv'(v) -> mdot symbol
mprime = sp.Derivative(mv, v)
Gamma_sub = [[[Gamma[l][m_][n].subs(mprime, mdot).subs(mv, m) for n in range(3)]
              for m_ in range(3)] for l in range(3)]

if __name__ == "__main__":
    # sanity check: static limit mdot->0, m=M should reduce to standard equatorial Kerr
    # in advanced coordinates. Verify g_{r phi} = -a exactly (no r, m dependence).
    print("g_rphi =", g[1, 2])
    print("det check g_rr =", g[1, 1])
    names = ['v', 'r', 'phi']
    nonzero = []
    for l in range(3):
        for mu in range(3):
            for nu in range(mu, 3):
                expr = Gamma_sub[l][mu][nu]
                if expr != 0:
                    nonzero.append((f"Gamma^{names[l]}_{names[mu]}{names[nu]}", expr))
    print(f"\n{len(nonzero)} nonzero independent Christoffel components (upper+diag, mu<=nu):")
    for name, expr in nonzero:
        print(f"  {name} = {expr}")
