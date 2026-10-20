"""
kvcore.py -- equatorial geodesics in the ingoing Kerr--Vaidya metric.

Units: initial mass M0 = 1 (G = c = 1).  Coordinates (v, r, phi), equatorial.
State y = [v, r, phi, p_v, p_r, p_phi]  (covariant momenta).

Hamiltonian H = 1/2 g^{mu nu} p_mu p_nu with the closed-form equatorial
inverse metric (verified symbolically in derive_identities.py):

  g^{vv}=a^2/r^2, g^{vr}=(r^2+a^2)/r^2, g^{v phi}=a/r^2,
  g^{rr}=Delta/r^2, g^{r phi}=a/r^2, g^{phi phi}=1/r^2,   Delta=r^2-2mr+a^2.

Hamilton's equations: dx/dtau = dH/dp, dp/dtau = -dH/dx; d/dv acts through
m(v) and a(v):  dp_v/dtau = -(mdot dH/dm + adot dH/da), with
  dH/dm = -p_r^2/r,   dH/da = (p_v+p_r)(a(p_v+p_r)+p_phi)/r^2.
For fixed a this gives the exact identity dE/dtau = -mdot p_r^2 / r (E=-p_v).

Mass-loss laws (par array):
  par = [law, alpha, dm, spin, M0, v_on]
  law = 0  ("A", fixed a):    a(v) = spin
  law = 1  ("B", fixed chi):  a(v) = spin * m(v)
  m(v) = M0                       for v <= v_on
       = M0 - alpha (v - v_on)    for v_on < v < v_on + dm/alpha
       = M0 - dm            afterwards (finite mass-loss episode).
"""
import numpy as np
from numba import njit

# ---------------------------------------------------------------- background
@njit(cache=True)
def mass_state(v, par):
    """return m, a, mdot, adot at advanced time v"""
    law, alpha, dm, spin, M0, von = par[0], par[1], par[2], par[3], par[4], par[5]
    if alpha <= 0.0 or v <= von:
        m = M0
        mdot = 0.0
    else:
        vend = von + dm / alpha
        if v < vend:
            m = M0 - alpha * (v - von)
            mdot = -alpha
        else:
            m = M0 - dm
            mdot = 0.0
    if law < 0.5:
        a = spin
        adot = 0.0
    else:
        a = spin * m
        adot = spin * mdot
    return m, a, mdot, adot


@njit(cache=True)
def r_plus(m, a):
    d = m * m - a * a
    if d < 0.0:
        return -1.0          # no horizon (naked singularity) -> flagged
    return m + np.sqrt(d)


@njit(cache=True)
def final_mass(par):
    if par[1] <= 0.0:
        return par[4]
    return par[4] - par[2]


@njit(cache=True)
def final_rplus(par):
    mf = final_mass(par)
    if par[0] < 0.5:
        af = par[3]
    else:
        af = par[3] * mf
    return r_plus(mf, af)


# ---------------------------------------------------------------- dynamics
@njit(cache=True)
def hamiltonian(y, par):
    v, r, ph, pv, pr, pp = y[0], y[1], y[2], y[3], y[4], y[5]
    m, a, md, ad = mass_state(v, par)
    D = r * r - 2.0 * m * r + a * a
    N = (a * a * pv * pv + 2.0 * (r * r + a * a) * pv * pr + 2.0 * a * pv * pp
         + D * pr * pr + 2.0 * a * pr * pp + pp * pp)
    return 0.5 * N / (r * r)


@njit(cache=True)
def rhs(y, par, out):
    v, r, ph, pv, pr, pp = y[0], y[1], y[2], y[3], y[4], y[5]
    m, a, md, ad = mass_state(v, par)
    r2 = r * r
    D = r2 - 2.0 * m * r + a * a
    out[0] = (a * a * pv + (r2 + a * a) * pr + a * pp) / r2
    out[1] = ((r2 + a * a) * pv + D * pr + a * pp) / r2
    out[2] = (a * pv + a * pr + pp) / r2
    N = (a * a * pv * pv + 2.0 * (r2 + a * a) * pv * pr + 2.0 * a * pv * pp
         + D * pr * pr + 2.0 * a * pr * pp + pp * pp)
    Nr = 4.0 * r * pv * pr + (2.0 * r - 2.0 * m) * pr * pr
    dHdr = 0.5 * Nr / r2 - N / (r2 * r)
    dHdm = -pr * pr / r
    s = pv + pr
    dHda = s * (a * s + pp) / r2
    out[3] = -(md * dHdm + ad * dHda)
    out[4] = -dHdr
    out[5] = 0.0


@njit(cache=True)
def energy_rate_identity(y, par):
    """analytic dE/dtau (E=-p_v): fixed a -> -mdot p_r^2/r ;
    law B adds -adot * dH/da with adot = chi mdot."""
    v, r, pv, pr, pp = y[0], y[1], y[3], y[4], y[5]
    m, a, md, ad = mass_state(v, par)
    s = pv + pr
    return -md * pr * pr / r + ad * s * (a * s + pp) / (r * r)


# ---------------------------------------------------------------- DOPRI5
C2, C3, C4, C5 = 1/5, 3/10, 4/5, 8/9
A21 = 1/5
A31, A32 = 3/40, 9/40
A41, A42, A43 = 44/45, -56/15, 32/9
A51, A52, A53, A54 = 19372/6561, -25360/2187, 64448/6561, -212/729
A61, A62, A63, A64, A65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
B1, B3, B4, B5, B6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
E1, E3, E4, E5, E6, E7 = (71/57600, -71/16695, 71/1920, -17253/339200,
                          22/525, -1/40)


@njit(cache=True)
def dp_step(y, h, par, k1):
    """one Dormand-Prince step from y with derivative k1 = f(y).
    returns y_new, err_vector, k7 (=f(y_new))"""
    n = 6
    k2 = np.empty(n); k3 = np.empty(n); k4 = np.empty(n)
    k5 = np.empty(n); k6 = np.empty(n); k7 = np.empty(n)
    yt = np.empty(n)
    for i in range(n):
        yt[i] = y[i] + h * A21 * k1[i]
    rhs(yt, par, k2)
    for i in range(n):
        yt[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i])
    rhs(yt, par, k3)
    for i in range(n):
        yt[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i])
    rhs(yt, par, k4)
    for i in range(n):
        yt[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i])
    rhs(yt, par, k5)
    for i in range(n):
        yt[i] = y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i]
                            + A64 * k4[i] + A65 * k5[i])
    rhs(yt, par, k6)
    ynew = np.empty(n)
    for i in range(n):
        ynew[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i]
                              + B5 * k5[i] + B6 * k6[i])
    rhs(ynew, par, k7)
    err = np.empty(n)
    for i in range(n):
        err[i] = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i]
                      + E6 * k6[i] + E7 * k7[i])
    return ynew, err, k7


@njit(cache=True)
def event_value(y, par, delta, mode):
    """mode 1: r - r_+(v) - delta (split location); mode 2: r - delta.
    Crossed downward."""
    if mode == 2:
        return y[1] - delta
    if mode == 3:                      # r - 2 m(v) - delta  (ergosurface crossing)
        m3, a3, md3, ad3 = mass_state(y[0], par)
        return y[1] - 2.0 * m3 - delta
    m, a, md, ad = mass_state(y[0], par)
    rp = r_plus(m, a)
    if rp < 0.0:
        return 1e9
    return y[1] - rp - delta


# status codes
RUNNING, CAPTURED, ESCAPED, EVENT, TIMEOUT, NAKED = 0, 1, 2, 3, 4, 5


@njit(cache=True)
def integrate(y0, par, tau_max, r_far, use_event, delta,
              rtol=1e-10, atol=1e-12, h0=1e-3, record=0):
    """Integrate a geodesic until capture / escape / event / timeout.

    captured : r < r_+(final) - 1e-3  (inside the final event horizon; the
               trapped region only shrinks to r_+(final), see paper)
    escaped  : r > r_far, dr/dtau > 0, and v beyond the end of the episode
               (so that E is constant thereafter)
    event    : r - r_+(v) - delta crosses zero from above (if use_event)
    naked    : m(v) < a(v) reached (no horizon)
    Returns (y, tau, status, rmin, traj, ntraj)."""
    y = y0.copy()
    k1 = np.empty(6)
    rhs(y, par, k1)
    tau = 0.0
    h = h0
    rcap = final_rplus(par) - 1e-3
    vend = 0.0
    if par[1] > 0.0:
        vend = par[5] + par[2] / par[1]
    rmin = y[1]
    nmax = record if record > 0 else 1
    traj = np.zeros((nmax, 7))
    ntraj = 0
    if record > 0:
        for i in range(6):
            traj[0, i] = y[i]
        traj[0, 6] = tau
        ntraj = 1
    fprev = event_value(y, par, delta, use_event)
    status = RUNNING
    nsteps = 0
    while True:
        nsteps += 1
        if nsteps > 5_000_000:
            status = TIMEOUT
            break
        if tau + h > tau_max:
            h = tau_max - tau
        # step-size cap: keep relative change of r moderate & resolve episode end
        hmax = 0.05 * y[1] + 0.01
        if h > hmax:
            h = hmax
        ynew, err, k7 = dp_step(y, h, par, k1)
        # error norm
        en = 0.0
        for i in range(6):
            sc = atol + rtol * max(abs(y[i]), abs(ynew[i]))
            q = err[i] / sc
            en += q * q
        en = np.sqrt(en / 6.0)
        if en > 1.0 or not np.isfinite(en):
            fac = 0.9 * en ** (-0.2) if np.isfinite(en) and en > 0 else 0.2
            if fac < 0.2:
                fac = 0.2
            h *= fac
            if h < 1e-14:
                status = TIMEOUT
                break
            continue
        # --- event detection (split location) -----------------------
        if use_event > 0:
            fnew = event_value(ynew, par, delta, use_event)
            if fprev > 0.0 and fnew <= 0.0:
                # bisection on the step length from y
                lo, hi = 0.0, h
                yb = ynew
                for it in range(80):
                    mid = 0.5 * (lo + hi)
                    ym, em, km = dp_step(y, mid, par, k1)
                    fm = event_value(ym, par, delta, use_event)
                    if fm > 0.0:
                        lo = mid
                    else:
                        hi = mid
                        yb = ym
                    if hi - lo < 1e-15 * (1.0 + tau):
                        break
                y = yb
                tau += hi
                status = EVENT
                break
            fprev = fnew
        y = ynew
        k1 = k7
        tau += h
        if y[1] < rmin:
            rmin = y[1]
        if record > 0 and ntraj < nmax:
            for i in range(6):
                traj[ntraj, i] = y[i]
            traj[ntraj, 6] = tau
            ntraj += 1
        m, a, md, ad = mass_state(y[0], par)
        if m < a:
            status = NAKED
            break
        if y[1] < rcap or y[1] < 0.05:
            status = CAPTURED
            break
        if y[1] > r_far and k1[1] > 0.0 and y[0] > vend:
            status = ESCAPED
            break
        if tau >= tau_max:
            status = TIMEOUT
            break
        # step growth
        fac = 0.9 * en ** (-0.2) if en > 0 else 5.0
        if fac > 5.0:
            fac = 5.0
        h *= fac
    return y, tau, status, rmin, traj, ntraj


# ---------------------------------------------------------------- utilities
@njit(cache=True)
def ginv_eq(r, m, a):
    G = np.empty((3, 3))
    r2 = r * r
    G[0, 0] = a * a / r2
    G[0, 1] = G[1, 0] = (r2 + a * a) / r2
    G[0, 2] = G[2, 0] = a / r2
    G[1, 1] = (r2 - 2 * m * r + a * a) / r2
    G[1, 2] = G[2, 1] = a / r2
    G[2, 2] = 1.0 / r2
    return G


@njit(cache=True)
def g_eq(r, m, a):
    G = np.empty((3, 3))
    G[0, 0] = -(1.0 - 2.0 * m / r)
    G[0, 1] = G[1, 0] = 1.0
    G[0, 2] = G[2, 0] = -2.0 * a * m / r
    G[1, 1] = 0.0
    G[1, 2] = G[2, 1] = -a
    G[2, 2] = r * r + a * a + 2.0 * m * a * a / r
    return G


def pr_from_shell(r, m, a, pv, pp, mu2, branch):
    """solve g^{mn}p_m p_n = -mu2 for p_r. branch 'in' -> dr/dtau<0 root."""
    G = ginv_eq(r, m, a)
    A = G[1, 1]
    B = 2 * (G[0, 1] * pv + G[1, 2] * pp)
    C = G[0, 0] * pv**2 + 2 * G[0, 2] * pv * pp + G[2, 2] * pp**2 + mu2
    if abs(A) < 1e-14:
        roots = [-C / B]
    else:
        disc = B * B - 4 * A * C
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        roots = [(-B - sq) / (2 * A), (-B + sq) / (2 * A)]
    best = None
    for prr in roots:
        drdtau = G[1, 0] * pv + G[1, 1] * prr + G[1, 2] * pp
        dvdtau = G[0, 0] * pv + G[0, 1] * prr + G[0, 2] * pp
        if dvdtau <= 0:          # must be future directed
            continue
        if branch == 'in' and drdtau < 0:
            best = prr
        if branch == 'out' and drdtau > 0:
            best = prr
    return best


def decay(y, par, beta, psi):
    """Decay of particle 1 (unit mass, covariant momentum in y) into two
    equal-mass fragments with rest-frame speed beta, emitted back-to-back
    along n(psi) = cos(psi) e1 + sin(psi) e2 in the equatorial rest-frame
    plane; e1 is the unit projection of the Killing vector d_v orthogonal to
    u (so psi=0 minimises the local energy of fragment 2).
    Four-momentum is conserved exactly: p1 = p2 + p3.
    Returns covariant momenta p2, p3 (3-vectors (p_v, p_r, p_phi))."""
    m, a, _, _ = mass_state(y[0], par)
    r = y[1]
    G = g_eq(r, m, a)
    Gi = ginv_eq(r, m, a)
    p1 = np.array([y[3], y[4], y[5]])
    u = Gi @ p1                                   # contravariant 4-velocity
    xi = np.array([1.0, 0.0, 0.0])
    w = np.array([0.0, 1.0, 0.0])
    dot = lambda A_, B_: A_ @ G @ B_
    xp = xi + dot(u, xi) * u
    e1 = xp / np.sqrt(dot(xp, xp))
    wp = w + dot(u, w) * u
    wp = wp - dot(wp, e1) * e1
    e2 = wp / np.sqrt(dot(wp, wp))
    n = np.cos(psi) * e1 + np.sin(psi) * e2
    p2c = 0.5 * (u + beta * n)
    p3c = 0.5 * (u - beta * n)
    return G @ p2c, G @ p3c


def local_bound(E1, r, m):
    """max of -E2/E1 over all directions for a beta=1 decay:
       1/2 (sqrt(1 + g_vv/E1^2) - 1),  g_vv = 2m/r - 1 (equatorial)."""
    gvv = 2 * m / r - 1
    return 0.5 * (np.sqrt(1 + gvv / E1**2) - 1)
