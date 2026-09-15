"""fig_trajectories.py -- run in kv_package/src/"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import geodesics as kv
from pipeline_v2 import run_case_v2
from pipeline import pr_from_mass_shell

def full_trajectory(y0, a, alpha, tau_max, r_plus_stop, r_esc=None):
    y0 = np.asarray(y0, dtype=float)
    if not np.all(np.isfinite(y0)):
        raise ValueError(f"Non-finite y0 detected: {y0}")

    def horizon_event(tau, y, a_, alpha_):
        return y[1] - (r_plus_stop + 0.005)
    horizon_event.terminal = True
    horizon_event.direction = -1

    events = [horizon_event]
    if r_esc is not None:
        def escape_event(tau, y, a_, alpha_):
            return y[1] - r_esc
        escape_event.terminal = True
        escape_event.direction = 1
        events.append(escape_event)

    return solve_ivp(
        kv.rhs, [0, tau_max], y0,
        args=(a, alpha),
        max_step=0.02,
        rtol=1e-9, atol=1e-11,
        events=events
    )

def safe_pr(r, a, m, E, L, direction):
    """Wrapper robusto intorno a pr_from_mass_shell."""
    try:
        pr = pr_from_mass_shell(r, a, m, E, L, direction)
    except Exception as e:
        print(f"[ERROR] pr_from_mass_shell raised: {e}")
        return None

    if pr is None:
        return None

    # Se restituisce una tupla/lista, prendiamo il primo elemento
    if isinstance(pr, (tuple, list, np.ndarray)):
        pr = pr[0]

    try:
        pr = complex(pr)
    except (TypeError, ValueError):
        print(f"[WARNING] impossibile convertire pr a complesso: {pr} (type={type(pr)})")
        return None

    if abs(pr.imag) > 1e-10:
        print(f"[WARNING] pr ha parte immaginaria non trascurabile: {pr}")
        return None

    pr = float(pr.real)
    if not np.isfinite(pr):
        return None
    return pr

def plot_case(ax, a, alpha, m0=1.0, R0=6.0, R_ESC=20.0, drmin=0.001,
              E_grid=None, L_grid=None, tau_max=400.0):
    kv.m0_global = m0
    res = run_case_v2(a, alpha, m0=m0, drmin=drmin,
                      E_grid=E_grid, L_grid=L_grid, tau_max=tau_max)
    if res is None:
        ax.set_title(f"a/M={a}, alpha={alpha}: NO VALID SPLIT")
        return None

    E1, L1 = res['E1'], res['L1']
    r_split = res['r_split']
    v_split = res['v_split']
    phi_split = res['phi_split']
    m_split = res['m_split']
    E2, L2 = res['E2'], res['L2']
    E3, L3 = res['E3'], res['L3']

    r_plus_0 = kv.r_plus(m0, a)

    # Incoming
    pr0 = safe_pr(R0, a, m0, E1, L1, 'in')
    if pr0 is None:
        ax.set_title(f"a/M={a}, alpha={alpha}: pr0 non-finite")
        return None

    sol_in = full_trajectory(
        [0.0, R0, 0.0, -E1, pr0, L1],
        a, alpha, tau_max, r_plus_0
    )
    r_in, phi_in = sol_in.y[1], sol_in.y[2]
    idx_split = np.argmin(np.abs(r_in - r_split))
    r_in, phi_in = r_in[:idx_split+1], phi_in[:idx_split+1]

    r_plus_here = kv.r_plus(m_split, a)

    # Fragment 2 (infalling)
    pr2 = safe_pr(r_split, a, m_split, E2, L2, 'in')
    if pr2 is None:
        print(f"[WARNING] pr2 non-finite → uso fallback. "
              f"E2={E2:.6f}, L2={L2:.6f}, r_split={r_split:.6f}, m_split={m_split:.6f}")
        pr2 = -1e-5

    sol2 = full_trajectory(
        [v_split, r_split, phi_split, -E2, pr2, L2],
        a, alpha, 100.0, r_plus_here
    )
    r2, phi2 = sol2.y[1], sol2.y[2]

    # Fragment 3 (escaping)
    pr3 = safe_pr(r_split, a, m_split, E3, L3, 'out')
    if pr3 is None:
        print(f"[WARNING] pr3 non-finite → uso fallback. "
              f"E3={E3:.6f}, L3={L3:.6f}")
        pr3 = +1e-5

    sol3 = full_trajectory(
        [v_split, r_split, phi_split, -E3, pr3, L3],
        a, alpha, 400.0, r_plus_here, r_esc=R_ESC
    )
    r3, phi3 = sol3.y[1], sol3.y[2]

    # Plot
    ax.plot(r_in * np.cos(phi_in), r_in * np.sin(phi_in),
            '-', color='tab:blue', label=f'Incoming ($E_1$={E1:.2f})')
    ax.plot(r2 * np.cos(phi2), r2 * np.sin(phi2),
            '--', color='tab:red', label=f'Fragment 2 ($E_2$={E2:.2f})')
    ax.plot(r3 * np.cos(phi3), r3 * np.sin(phi3),
            '-.', color='tab:green', label=f'Fragment 3 ($E_3$={E3:.2f})')
    ax.plot(r_split * np.cos(phi_split), r_split * np.sin(phi_split),
            'kx', markersize=10, label='Split point')

    theta = np.linspace(0, 2 * np.pi, 200)
    rs = kv.r_s(m_split)
    ax.fill(rs * np.cos(theta), rs * np.sin(theta), color='orange', alpha=0.2)
    ax.fill(r_plus_here * np.cos(theta), r_plus_here * np.sin(theta),
            color='black', alpha=0.8)

    ax.set_aspect('equal')
    ax.set_title(f"a/M={a}, alpha={alpha}\n"
                 + r"$\eta$" + f"={res['eta']*100:.1f}%, "
                 + r"$\Delta E$" + f"={res['dE']:.3f}M")
    ax.legend(fontsize=7, loc='upper right')
    return res

# ------------------------------------------------------------------
if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    cases = [
        (0.9, 0.0, {}),
        (0.9, 5e-4, dict(
            E_grid=np.linspace(0.5, 1.05, 24),
            L_grid=np.linspace(-8, 8, 36),
            tau_max=400.0
        ))
    ]

    for col, (a, alpha, kwargs) in enumerate(cases):
        # riga superiore: vista completa
        plot_case(axes[0, col], a, alpha, **kwargs)

        # riga inferiore: stesso caso, zoom vicino all'orizzonte
        plot_case(axes[1, col], a, alpha, **kwargs)
        axes[1, col].set_xlim(-3, 3)
        axes[1, col].set_ylim(-3, 3)
        axes[1, col].set_title(axes[1, col].get_title() + "\n(zoom near horizon)")

    fig.suptitle("Optimal Penrose trajectories: static vs dynamic Kerr-Vaidya (corrected)")
    fig.tight_layout()
    fig.savefig("fig_trajectories_static_vs_dynamic.pdf", bbox_inches="tight")
    fig.savefig("fig_trajectories_static_vs_dynamic.png", dpi=150, bbox_inches="tight")
    print("Saved fig_trajectories_static_vs_dynamic.pdf + .png")
    plt.show()