"""Independent analytic test: radial null geodesics in ingoing Vaidya (a=0).
Outgoing radial null rays obey dr/dv = (1 - 2 m(v)/r)/2 exactly; we compare
r(v) from the Hamiltonian integrator with an independent solution of this ODE."""
import numpy as np
from scipy.integrate import solve_ivp
import kvcore as K
out = []
for alpha in (1e-3, 1e-2, 1e-1):
    P = np.array([0.0, alpha, 0.05, 0.0, 1.0, 0.0])
    r0 = 2.5
    # outgoing radial photon: p_phi=0, choose p_v=-1, p_r from null shell with dr/dtau>0
    pr = K.pr_from_shell(r0, 1.0, 0.0, -1.0, 0.0, 0.0, 'out')
    y0 = np.array([0.0, r0, 0.0, -1.0, pr, 0.0])
    _, _, _, _, tr, n = K.integrate(y0, P, 1e4, 60.0, 0, 0.0, 1e-12, 1e-14, 1e-4, 2_000_000)
    tr = tr[:n]
    m = lambda v: max(1.0 - alpha * max(v, 0.0), 0.95)
    ref = solve_ivp(lambda v, r: 0.5 * (1 - 2 * m(v) / r), [0, tr[-1, 0]], [r0],
                    rtol=1e-12, atol=1e-14, dense_output=True, max_step=0.01)
    err = np.max(np.abs(ref.sol(tr[:, 0])[0] - tr[:, 1]) / tr[:, 1])
    out.append(f"alpha={alpha:g}: max relative deviation of r(v) = {err:.1e} over v in [0,{tr[-1,0]:.1f}]")
    print(out[-1])
open('../data/vaidya_test.txt', 'w').write("\n".join(out) + "\n")
