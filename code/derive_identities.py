"""
Symbolic derivations for the Kerr--Vaidya Penrose paper.

Metric: ingoing (advanced) Kerr--Vaidya, coordinates (v, r, theta, phi),
mass m(v) and (optionally) spin parameter a(v):

 ds^2 = -(1 - 2 m r/Sigma) dv^2 + 2 dv dr - 2 a s^2 dr dphi
        - (4 a m r s^2/Sigma) dv dphi + Sigma dtheta^2
        + [(r^2+a^2)^2 - Delta a^2 s^2] s^2/Sigma dphi^2,
 Sigma = r^2 + a^2 cos^2 theta,  Delta = r^2 - 2 m r + a^2,  s = sin theta.

Results checked here:
 (1) d p_v / d tau = 1/2 (d_v g_{mu nu}) p^mu p^nu   (geodesic, covariant form)
     For a = const:  1/2 d_v g p p = mdot * (r/Sigma) * (l.p)^2,
     l_mu = (-1, 0, 0, a s^2)  (principal null covector of the flux).
 (2) l is null; T_{mu nu} of the metric (equatorial check) is prop. to mdot l l
     at leading order (we only verify the null property and l^mu components).
 (3) For a = chi m(v): additional term; printed in simplified form.
 (4) Equatorial inverse metric, for the numerical code.
 (5) Local decay bound: max over unit n orthogonal to u of n.xi equals
     sqrt(E^2 + g_vv); reproduces BPT at the horizon for E=1.
"""
import sympy as sp

v, r, th, ph = sp.symbols('v r theta phi', real=True)
M, A = sp.symbols('m a', positive=True)          # instantaneous values
mdot, adot = sp.symbols('mdot adot', real=True)
pv, pr, pth, pph = sp.symbols('p^v p^r p^theta p^phi', real=True)

s = sp.sin(th)
c = sp.cos(th)
Sigma = r**2 + A**2*c**2
Delta = r**2 - 2*M*r + A**2

def metric(m, a):
    Sig = r**2 + a**2*c**2
    Del = r**2 - 2*m*r + a**2
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2*m*r/Sig)
    g[0, 1] = g[1, 0] = 1
    g[0, 3] = g[3, 0] = -2*a*m*r*s**2/Sig
    g[1, 3] = g[3, 1] = -a*s**2
    g[2, 2] = Sig
    g[3, 3] = ((r**2 + a**2)**2 - Del*a**2*s**2)*s**2/Sig
    return g

g = metric(M, A)
P = sp.Matrix([pv, pr, pth, pph])

# --- (1) energy-transport identity -------------------------------------
dg_dm = g.applyfunc(lambda e: sp.diff(e, M))
dg_da = g.applyfunc(lambda e: sp.diff(e, A))
quad_m = sp.simplify((P.T*dg_dm*P)[0])
quad_a = sp.simplify((P.T*dg_da*P)[0])

l_cov = sp.Matrix([-1, 0, 0, A*s**2])
lp = (l_cov.T*P)[0]
claim = 2*r/Sigma*lp**2
print("(1) d_m g_{mn} p^m p^n - (2 r/Sigma)(l.p)^2 =",
      sp.simplify(quad_m - claim))
# => dp_v/dtau = 1/2 mdot quad_m = mdot (r/Sigma)(l.p)^2 ;  E=-p_v
print("    hence dE/dtau = -mdot (r/Sigma) (l.p)^2   [fixed a]")

# --- (2) l null, l^mu ----------------------------------------------------
ginv = sp.simplify(g.inv())
l_up = sp.simplify(ginv*l_cov)
print("(2) l.l =", sp.simplify((l_cov.T*ginv*l_cov)[0]), "; l^mu =", list(l_up))

# --- (3) a = chi m law: d/dv = mdot d/dm + adot d/da with adot = chi mdot
chi = sp.symbols('chi', positive=True)
quad_a_eq = sp.simplify(quad_a.subs(th, sp.pi/2))
print("(3) equatorial d_a g_{mn}p^m p^n =", sp.factor(quad_a_eq))
lp_eq = lp.subs(th, sp.pi/2)
# try to write it compactly
alt = 2*pph*(-(2*M/r)*(pv - A*pph) - pr + A*pph)
print("    check compact form:", sp.simplify(quad_a_eq - alt))

# --- (4) equatorial inverse metric for numerics (v, r, phi) ---------------
g3 = metric(M, A).subs(th, sp.pi/2)
idx = [0, 1, 3]
g3 = sp.Matrix(3, 3, lambda i, j: g3[idx[i], idx[j]])
g3inv = sp.simplify(g3.inv())
print("(4) equatorial g^{mu nu} (v,r,phi):")
for i in range(3):
    for j in range(i, 3):
        print("   ", i, j, sp.factor(g3inv[i, j]))
print("    det g3 =", sp.simplify(g3.det()))

# --- (5) local decay bound ------------------------------------------------
# For unit spacelike n orthogonal to u (u.u=-1), max n.xi = |xi_perp|,
# xi_perp = xi + (u.xi) u,  xi_perp^2 = xi.xi + (u.xi)^2 = g_vv + E^2.
E = sp.symbols('E', positive=True)
gvv_eq = sp.simplify(g3[0, 0])
rp = M + sp.sqrt(M**2 - A**2)
bound = sp.Rational(1, 2)*(sp.sqrt(E**2 + gvv_eq) - E)/E
bpt = bound.subs({E: 1, r: rp})
print("(5) eta_loc_max(E=1, r=r+) =", sp.simplify(bpt),
      " ; BPT form (1/2)(sqrt(2m/r+)-1):",
      sp.simplify(bpt - sp.Rational(1, 2)*(sp.sqrt(2*M/rp) - 1)))
print("    extremal a=m:", sp.nsimplify(sp.simplify(bpt.subs(A, M))))

# --- horizon / ergosphere -------------------------------------------------
print("    g^{rr} =", sp.factor(g3inv[1, 1]), "-> zero at Delta=0")
