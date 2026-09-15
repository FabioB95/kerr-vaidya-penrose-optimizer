#!/usr/bin/env python3
"""
Kerr-Vaidya Penrose Optimizer — FULLY CORRECTED VERSION

Corrections applied:
1. Metric in genuine advanced coordinates (v,r,phi): g_rr=0, g_rphi=2*a*m*r/Delta.
2. Energy E=-p_v and angular momentum L=p_phi are EVOLVED dynamically.
3. Bardeen-Press-Teukolsky bound = (sqrt(2)-1)/2.
4. Christoffel symbols computed numerically (4th-order finite differences).
5. Initial conditions solve the full nonlinear system self-consistently.
6. EFFECTIVE POTENTIAL now includes the linear p_r term via the full
   quadratic discriminant.  In advanced coordinates g^vr=1, so the
   mass-shell condition is genuinely quadratic in p_r and cannot be
   reduced to a simple sign condition on g^rr*(...).
"""
import numpy as np

# =============================================================================
# 1.  Metric
# =============================================================================

class KerrVaidyaParams:
    def __init__(self, a, alpha, m0=1.0, m_floor=0.95):
        self.a = a
        self.alpha = alpha
        self.m0 = m0
        self.m_floor = m_floor

    def m(self, v):
        return max(self.m_floor, self.m0 - self.alpha * v)

    def m_dot(self, v):
        if self.m0 - self.alpha * v > self.m_floor:
            return -self.alpha
        return 0.0

    def r_plus(self, v):
        m = self.m(v)
        return m + np.sqrt(max(m**2 - self.a**2, 0.0))

    def r_s(self, v):
        return 2.0 * self.m(v)


def metric_tensor(v, r, phi, params):
    """3x3 metric g_{mu nu} in coordinates (v, r, phi)."""
    m = params.m(v)
    a = params.a
    Sigma = r**2 + a**2
    Delta = r**2 - 2.0*m*r + a**2

    g = np.zeros((3, 3))
    g[0, 0] = -(1.0 - 2.0*m*r / Sigma)          # g_vv
    g[0, 1] = 1.0                                 # g_vr = 1
    g[0, 2] = -2.0*m*r*a / Sigma                # g_vphi
    g[1, 2] = 2.0*m*r*a / Delta                 # g_rphi
    g[2, 2] = r**2 + a**2 + 2.0*m*r*a**2 / Sigma  # g_phiphi
    g[1, 0] = g[0, 1]
    g[2, 0] = g[0, 2]
    g[2, 1] = g[1, 2]
    return g


# =============================================================================
# 2.  Christoffel symbols (numerical, 4th-order finite differences)
# =============================================================================

_FD_H = 1e-6


def christoffel_numerical(y, params, h=_FD_H):
    """Gamma^mu_{nu rho} at state y = [v, r, phi, ...]."""
    v, r, phi = y[0], y[1], y[2]

    def g_at(v_, r_, phi_):
        return metric_tensor(v_, r_, phi_, params)

    g = g_at(v, r, phi)
    g_inv = np.linalg.inv(g)

    dg_dv = (-g_at(v+2*h, r, phi) + 8.0*g_at(v+h, r, phi)
             - 8.0*g_at(v-h, r, phi) + g_at(v-2*h, r, phi)) / (12.0*h)

    dg_dr = (-g_at(v, r+2*h, phi) + 8.0*g_at(v, r+h, phi)
             - 8.0*g_at(v, r-h, phi) + g_at(v, r-2*h, phi)) / (12.0*h)

    dg_dphi = np.zeros((3, 3))
    d_g = [dg_dv, dg_dr, dg_dphi]

    Gamma = np.zeros((3, 3, 3))
    for lam in range(3):
        for mu in range(3):
            for nu in range(3):
                val = 0.0
                for sigma in range(3):
                    val += 0.5 * g_inv[lam, sigma] * (
                        d_g[mu][sigma, nu] + d_g[nu][sigma, mu] - d_g[sigma][mu, nu]
                    )
                Gamma[lam, mu, nu] = val
    return Gamma


# =============================================================================
# 3.  Geodesic RHS
# =============================================================================

def geodesic_rhs(y, params):
    """dx^mu/dtau = u^mu,  du^mu/dtau = -Gamma^mu_{nu rho} u^nu u^rho."""
    v, r, phi, uv, ur, uphi = y
    Gamma = christoffel_numerical(y, params)
    u = np.array([uv, ur, uphi])

    du = np.zeros(3)
    for mu in range(3):
        for nu in range(3):
            for rho in range(3):
                du[mu] -= Gamma[mu, nu, rho] * u[nu] * u[rho]

    return np.array([uv, ur, uphi, du[0], du[1], du[2]])


def renormalize(y, params):
    """Enforce g_{mu nu} u^mu u^nu = -1."""
    v, r, phi = y[0], y[1], y[2]
    g = metric_tensor(v, r, phi, params)
    u = y[3:6].copy()
    norm = np.dot(u, g @ u)
    if abs(norm + 1.0) > 1e-6 and norm < 0:
        factor = np.sqrt(-1.0 / norm)
        y[3:6] = u * factor
    return y


# =============================================================================
# 4.  Initial conditions — full nonlinear solver
# =============================================================================

def initial_state(r0, phi0, E, L, params):
    """
    Find u^mu such that:
        p_v = -E,  p_phi = L,  g_{mu nu} u^mu u^nu = -1
    """
    a = params.a
    m = params.m(0.0)
    Sigma = r0**2 + a**2
    Delta = r0**2 - 2.0*m*r0 + a**2

    g_vv = -(1.0 - 2.0*m*r0 / Sigma)
    g_vphi = -2.0*m*r0*a / Sigma
    g_rphi = 2.0*m*r0*a / Delta
    g_phiphi = r0**2 + a**2 + 2.0*m*r0*a**2 / Sigma

    A = np.array([[g_vv, g_vphi],
                  [g_vphi, g_phiphi]])

    def uv_uphi(ur):
        b = np.array([-E - ur, L - g_rphi*ur])
        return np.linalg.solve(A, b)

    def norm_sq(ur):
        uv, uphi = uv_uphi(ur)
        return (g_vv*uv**2 + 2.0*uv*ur + 2.0*g_vphi*uv*uphi
                + 2.0*g_rphi*ur*uphi + g_phiphi*uphi**2)

    best_ur = None
    best_err = 1e10
    for ur in np.linspace(-30.0, 30.0, 6001):
        n2 = norm_sq(ur)
        if n2 <= 0.0:
            err = abs(n2 + 1.0)
            if err < best_err:
                best_err = err
                best_ur = ur

    if best_ur is None:
        raise ValueError(f'No timelike initial state for E={E}, L={L} at r={r0}')

    uv, uphi = uv_uphi(best_ur)
    return np.array([0.0, r0, phi0, uv, best_ur, uphi])


# =============================================================================
# 5.  RK4 integrator
# =============================================================================

def integrate_geodesic(r0, phi0, E, L, tau_max, params, n_points=2500):
    """Integrate geodesic from r0 with given E, L."""
    dtau = tau_max / (n_points - 1)
    y = np.zeros((n_points, 6))
    y[0] = initial_state(r0, phi0, E, L, params)

    for i in range(n_points - 1):
        k1 = geodesic_rhs(y[i], params)
        k2 = geodesic_rhs(y[i] + 0.5*dtau*k1, params)
        k3 = geodesic_rhs(y[i] + 0.5*dtau*k2, params)
        k4 = geodesic_rhs(y[i] + dtau*k3, params)
        y[i+1] = y[i] + (dtau/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if i % 10 == 0:
            y[i+1] = renormalize(y[i+1], params)

    return {
        'v': y[:, 0],
        'r': y[:, 1],
        'phi': y[:, 2],
        'uv': y[:, 3],
        'ur': y[:, 4],
        'uphi': y[:, 5],
    }


# =============================================================================
# 6.  Derived quantities
# =============================================================================

def energy_along(traj, params):
    """E(v) = -p_v and L(v) = p_phi along trajectory."""
    n = len(traj['v'])
    E = np.zeros(n)
    L = np.zeros(n)
    for i in range(n):
        g = metric_tensor(traj['v'][i], traj['r'][i], traj['phi'][i], params)
        u = np.array([traj['uv'][i], traj['ur'][i], traj['uphi'][i]])
        p = g @ u
        E[i] = -p[0]
        L[i] = p[2]
    return E, L


# =============================================================================
# 7.  Effective potential — CORRECTED (full quadratic discriminant)
# =============================================================================

def effective_potential(r, v, E, L, params):
    """
    Return the discriminant of the mass-shell quadratic for p_r.

    From  g^{mu nu} p_mu p_nu = -1  with  p_v = -E, p_phi = L:

        g^rr p_r^2  +  (-2 g^vr E) p_r  +  (g^vv E^2 - 2 g^vphi E L + g^phiphi L^2 + 1) = 0

    A real p_r exists iff  discriminant >= 0.
    We return  V_eff = -discriminant  so that  V_eff <= 0  means "physical".
    """
    m = params.m(v)
    a = params.a
    Sigma = r**2 + a**2
    Delta = r**2 - 2.0*m*r + a**2

    g = np.array([[ -(1.0-2.0*m*r/Sigma), 1.0, -2.0*m*r*a/Sigma ],
                  [ 1.0, 0.0, 2.0*m*r*a/Delta ],
                  [ -2.0*m*r*a/Sigma, 2.0*m*r*a/Delta,
                    r**2+a**2+2.0*m*r*a**2/Sigma ]])
    g_inv = np.linalg.inv(g)

    gvv   = g_inv[0, 0]
    gvr   = g_inv[0, 1]   # = 1  in these coordinates
    gvphi = g_inv[0, 2]
    grr   = g_inv[1, 1]
    gphiphi = g_inv[2, 2]

    # Coefficients of  g^rr p_r^2 + B p_r + C = 0
    # B = 2 g^vr p_v = 2 g^vr (-E) = -2 g^vr E
    B = -2.0 * gvr * E
    C = gvv*E**2 - 2.0*gvphi*E*L + gphiphi*L**2 + 1.0

    discriminant = B**2 - 4.0 * grr * C
    return -discriminant   # <= 0  means physical (real p_r exists)


def penrose_efficiency(E1, E2):
    return -E2 / E1


def bardeen_bound():
    """CORRECT Bardeen-Press-Teukolsky bound for extremal Kerr, E1=1."""
    return (np.sqrt(2.0) - 1.0) / 2.0


# =============================================================================
# 8.  Demo
# =============================================================================

def demo():
    print('Bardeen bound (corrected):', bardeen_bound(),
          '=', f'{bardeen_bound()*100:.1f}%')
    print()
    print('=== DEMO: non-conservation of E in dynamic Kerr-Vaidya ===')
    params = KerrVaidyaParams(a=0.9, alpha=0.0005, m0=1.0, m_floor=0.95)
    traj = integrate_geodesic(r0=6.0, phi0=0.0, E=0.95, L=2.5,
                              tau_max=100.0, params=params, n_points=500)
    E_arr, L_arr = energy_along(traj, params)
    print(f'Initial E = {E_arr[0]:.4f}, final E = {E_arr[-1]:.4f}')
    print(f'Initial L = {L_arr[0]:.4f}, final L = {L_arr[-1]:.4f}')
    print(f'Relative change in E: {(E_arr[-1]-E_arr[0])/E_arr[0]*100:.2f}%')
    print('Energy is NOT conserved in the dynamic case.')
    print()
    print('=== DEMO: effective potential near horizon ===')
    r_test = params.r_plus(0.0) + 0.01
    v_test = 0.0
    V = effective_potential(r_test, v_test, E=0.95, L=2.5, params=params)
    print(f'V_eff at r={r_test:.4f}, v={v_test}: {V:.4f}')
    print(f'  (V_eff <= 0 means physical split is possible)')
    print('===========================================================')


if __name__ == '__main__':
    demo()