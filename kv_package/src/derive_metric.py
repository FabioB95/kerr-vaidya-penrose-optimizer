"""
Step 1: derive the Kerr metric in ingoing ('advanced') coordinates
(v, r, theta, phi) by transforming from Boyer-Lindquist (t, r, theta, phi),
to get an independently-verified ground truth for g_{r phi} etc.

Transformation (standard, e.g. Visser 2007 arXiv:0706.0622 Sec. 4):
    dv   = dt + (r^2+a^2)/Delta dr
    dphi' = dphi + a/Delta dr
theta unchanged.
"""
import sympy as sp

t, r, th, ph, M, a = sp.symbols('t r theta phi M a', real=True)
Sigma = r**2 + a**2*sp.cos(th)**2
Delta = r**2 - 2*M*r + a**2

# Boyer-Lindquist Kerr metric components (t,r,theta,phi)
g_bl = sp.zeros(4, 4)
g_bl[0, 0] = -(1 - 2*M*r/Sigma)
g_bl[1, 1] = Sigma/Delta
g_bl[2, 2] = Sigma
g_bl[3, 3] = (r**2 + a**2 + 2*M*r*a**2*sp.sin(th)**2/Sigma) * sp.sin(th)**2
g_bl[0, 3] = g_bl[3, 0] = -2*M*r*a*sp.sin(th)**2/Sigma

# coordinate change: t = v - f(r), phi = phi' - g(r), with
# df/dr = (r^2+a^2)/Delta,  dg/dr = a/Delta
# so dt = dv - f'(r) dr,  dphi = dphi' - g'(r) dr
v, phip = sp.symbols('v phi_p', real=True)
fprime = (r**2 + a**2)/Delta
gprime = a/Delta

old = [t, r, th, ph]
# Jacobian dx^old_mu / dx^new_nu, new coords = (v, r, theta, phi_p)
J = sp.Matrix([
    [1, -fprime, 0, 0],   # dt = dv - f' dr
    [0, 1, 0, 0],         # dr = dr
    [0, 0, 1, 0],         # dtheta = dtheta
    [0, -gprime, 0, 1],   # dphi = dphi_p - g' dr
])

g_new = sp.simplify(J.T * g_bl * J)
print("g_vv  =", sp.simplify(g_new[0, 0]))
print("g_vr  =", sp.simplify(g_new[0, 1]))
print("g_vth =", sp.simplify(g_new[0, 2]))
print("g_vph =", sp.simplify(g_new[0, 3]))
print("g_rr  =", sp.simplify(g_new[1, 1]))
print("g_rph =", sp.simplify(g_new[1, 3]))
print("g_thth=", sp.simplify(g_new[2, 2]))
print("g_phph=", sp.simplify(g_new[3, 3]))
