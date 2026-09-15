#!/usr/bin/env python3
"""
================================================================================
FILE: src/step03_dynamic_ergosphere_kv.py
================================================================================

Step 03: Dynamic Kerr-Vaidya — The Shrinking Ergosphere (ESCAPE-VERIFIED)
=========================================================================

This version fixes the critical issue from the previous attempt:
the split search now ACTUALLY INTEGRATES fragment 3 and verifies
that it escapes to r > 5M before accepting the split.

Only physically realizable Penrose processes are reported.

Outputs:
    figures/step03_dynamic_vs_static_ergosphere.png
    figures/step03_dynamic_vs_static_ergosphere.pdf

Run:
    python src/step03_dynamic_ergosphere_kv.py
================================================================================
"""

import sys
import os
import warnings

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import matplotlib.pyplot as plt
from kerr_vaidya_metric import (
    KerrVaidyaParams,
    integrate_geodesic,
    event_horizon_radius,
    effective_potential,
    penrose_efficiency,
)

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

M0 = 1.0
A = 0.9
ALPHA = 0.002
M_FLOOR = 0.5

def m_func_static(t):
    return M0

def m_dot_func_static(t):
    return 0.0

def m_func_dynamic(t):
    return max(M_FLOOR, M0 - ALPHA * t)

def m_dot_func_dynamic(t):
    m = M0 - ALPHA * t
    return -ALPHA if m > M_FLOOR else 0.0

PARAMS_STATIC = KerrVaidyaParams(a=A, m_func=m_func_static, m_dot_func=m_dot_func_static)
PARAMS_DYNAMIC = KerrVaidyaParams(a=A, m_func=m_func_dynamic, m_dot_func=m_dot_func_dynamic)

R0 = 6.0
PHI0 = 0.0
E1 = 0.95
L1 = 1.50
P_R0 = -0.50

TAU_MAX = 400.0
N_POINTS = 2000

OUT_PNG = os.path.join(project_root, 'figures', 'step03_dynamic_vs_static_ergosphere.png')
OUT_PDF = os.path.join(project_root, 'figures', 'step03_dynamic_vs_static_ergosphere.pdf')
DPI = 300

# ==============================================================================
# Helper: Find ergosphere crossing (dynamic-aware)
# ==============================================================================

def find_ergosphere_crossing(traj, m_func, a):
    for i, (t, r) in enumerate(zip(traj['t'], traj['r'])):
        m = m_func(t)
        r_s = 2.0 * m
        if r <= r_s:
            return i
    return None


def find_valid_split_with_escape(r_split, t_split, m_func, a, E1, L1,
                                 n_samples_e2=40, n_samples_l2=40,
                                 escape_threshold=5.0):
    """
    Search for (E2, L2) such that:
    1. V_eff <= 0 for both fragments at split point
    2. Fragment 3 actually escapes to r > escape_threshold
    3. We maximize efficiency among valid candidates
    """
    m = m_func(t_split)
    best = None
    best_eta = -1.0

    E2_vals = np.linspace(-0.45, -0.03, n_samples_e2)
    L2_vals = np.linspace(0.05, L1 - 0.05, n_samples_l2)

    total = len(E2_vals) * len(L2_vals)
    tested = 0

    for E2 in E2_vals:
        E3 = E1 - E2
        if E3 <= 0:
            continue
        for L2 in L2_vals:
            tested += 1
            L3 = L1 - L2
            if L3 <= 0:
                continue

            # Check effective potential at split point
            V2 = effective_potential(r_split, m, a, E2, L2)
            V3 = effective_potential(r_split, m, a, E3, L3)
            if V2 > 0 or V3 > 0:
                continue

            # NOW: actually integrate fragment 3 and check escape
            # Try a range of outward radial momenta
            escaped = False
            best_pr = None
            best_final_r = 0.0

            for p_r3 in np.linspace(0.1, 2.0, 15):
                try:
                    traj3 = integrate_geodesic(
                        r0=r_split, phi0=0.0,  # phi0 doesn't matter for escape check
                        E=E3, L=L3, p_r0=p_r3,
                        tau_max=300.0,
                        params=KerrVaidyaParams(a=a, m_func=m_func, m_dot_func=m_func_dynamic if m_func != m_func_static else m_dot_func_static),
                        n_points=300
                    )
                except Exception:
                    continue

                final_r = traj3['r'][-1]
                if final_r > escape_threshold:
                    escaped = True
                    if final_r > best_final_r:
                        best_final_r = final_r
                        best_pr = p_r3
                    break  # Found escape for this (E2,L2)

            if escaped:
                eta = penrose_efficiency(E1, E2, E3)
                if eta > best_eta:
                    best_eta = eta
                    best = {
                        'E2': E2, 'L2': L2,
                        'E3': E3, 'L3': L3,
                        'eta': eta,
                        'p_r3': best_pr,
                        'final_r3': best_final_r,
                        'V2': V2, 'V3': V3
                    }

    return best


def integrate_post_split_verified(r_split, phi_split, t_split, E2, L2, E3, L3, p_r3, params, label):
    """Integrate both fragments with verified escape parameters."""

    # Fragment 2: negative energy, falls in
    traj2 = integrate_geodesic(
        r0=r_split, phi0=phi_split,
        E=E2, L=L2, p_r0=-0.1,
        tau_max=100.0,
        params=params,
        n_points=500
    )

    # Fragment 3: use the verified p_r3
    traj3 = integrate_geodesic(
        r0=r_split, phi0=phi_split,
        E=E3, L=L3, p_r0=p_r3,
        tau_max=500.0,
        params=params,
        n_points=1500
    )

    print(f"    [{label}] Fragment 2 final r = {traj2['r'][-1]:.3f}M")
    print(f"    [{label}] Fragment 3 final r = {traj3['r'][-1]:.3f}M (p_r0={p_r3:.2f})")

    return traj2, traj3


def plot_ergosphere_snapshot(ax, m, a, alpha=0.15, color_ergo='#FF6B00', color_horizon='#1A1A1A'):
    theta = np.linspace(0, 2 * np.pi, 200)
    r_s = 2.0 * m
    r_plus = m + np.sqrt(max(m**2 - a**2, 0.0))

    x_ergo = r_s * np.cos(theta)
    y_ergo = r_s * np.sin(theta)
    x_horizon = r_plus * np.cos(theta)
    y_horizon = r_plus * np.sin(theta)

    ax.fill(x_ergo, y_ergo, alpha=alpha, color=color_ergo)
    ax.plot(x_ergo, y_ergo, color=color_ergo, linewidth=1.0, alpha=0.6)
    ax.fill(x_horizon, y_horizon, alpha=0.4, color=color_horizon)
    ax.plot(x_horizon, y_horizon, color=color_horizon, linewidth=1.5)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':

    print("=" * 70)
    print("Step 03: Dynamic Kerr-Vaidya — Escape-Verified Penrose Process")
    print("=" * 70)

    results = {}

    for label, params in [('Static', PARAMS_STATIC), ('Dynamic', PARAMS_DYNAMIC)]:

        print(f"\n{'-' * 60}")
        print(f"Case: {label}")
        print(f"{'-' * 60}")

        # ---------------------------------------------------------------------
        # 1. Incoming trajectory
        # ---------------------------------------------------------------------
        print(f"\n[1] Integrating incoming geodesic ({label})...")
        traj = integrate_geodesic(
            r0=R0, phi0=PHI0, E=E1, L=L1, p_r0=P_R0,
            tau_max=TAU_MAX, params=params, n_points=N_POINTS
        )
        print(f"    Status: {traj['message']}")
        print(f"    Final radius: r = {traj['r'][-1]:.4f}M")
        print(f"    Final time: t = {traj['t'][-1]:.2f}M")

        # ---------------------------------------------------------------------
        # 2. Ergosphere crossing
        # ---------------------------------------------------------------------
        print(f"\n[2] Locating ergosphere entry ({label})...")
        cross_idx = find_ergosphere_crossing(traj, params.m_func, A)

        if cross_idx is None:
            print(f"    ERROR: No ergosphere entry!")
            continue

        t_cross = traj['t'][cross_idx]
        r_cross = traj['r'][cross_idx]
        phi_cross = traj['phi'][cross_idx]
        m_cross = params.m_func(t_cross)
        r_s_cross = 2.0 * m_cross
        r_plus_cross = m_cross + np.sqrt(max(m_cross**2 - A**2, 0.0))

        print(f"    Entry: t={t_cross:.2f}M, r={r_cross:.4f}M, phi={phi_cross:.4f}")
        print(f"    Mass: m={m_cross:.4f}M, r_s={r_s_cross:.4f}M, r_+={r_plus_cross:.4f}M")

        # ---------------------------------------------------------------------
        # 3. Find escape-verified split
        # ---------------------------------------------------------------------
        print(f"\n[3] Searching for escape-verified Penrose split ({label})...")
        print(f"    This may take 20-60 seconds...")

        split = find_valid_split_with_escape(
            r_cross, t_cross, params.m_func, A, E1, L1,
            n_samples_e2=35, n_samples_l2=35
        )

        if split is None:
            print(f"    ERROR: No escape-verified split found in {label} case!")
            print(f"    The particle enters the ergosphere but no physically realizable")
            print(f"    Penrose split exists for these initial conditions.")
            continue

        E2 = split['E2']
        L2 = split['L2']
        E3 = split['E3']
        L3 = split['L3']
        p_r3 = split['p_r3']
        eta = split['eta']

        print(f"    FOUND valid split:")
        print(f"    E2={E2:.4f}, L2={L2:.4f}, V2={split['V2']:.4f}")
        print(f"    E3={E3:.4f}, L3={L3:.4f}, V3={split['V3']:.4f}")
        print(f"    p_r3={p_r3:.2f}, final_r3={split['final_r3']:.2f}M")
        print(f"    Efficiency: eta = {eta*100:.2f}%")

        # ---------------------------------------------------------------------
        # 4. Post-split integration
        # ---------------------------------------------------------------------
        print(f"\n[4] Post-split integration ({label})...")

        traj2, traj3 = integrate_post_split_verified(
            r_cross, phi_cross, t_cross,
            E2, L2, E3, L3, p_r3,
            params, label
        )

        # Verify
        if traj3['r'][-1] <= 5.0:
            print(f"    WARNING: Fragment 3 did not escape in high-res run!")
            continue

        # ---------------------------------------------------------------------
        # 5. Store
        # ---------------------------------------------------------------------
        results[label] = {
            'traj': traj, 'cross_idx': cross_idx,
            'traj2': traj2, 'traj3': traj3,
            'E2': E2, 'E3': E3, 'eta': eta,
            'm_cross': m_cross, 'r_s_cross': r_s_cross,
            'r_plus_cross': r_plus_cross, 'p_r3': p_r3,
        }

    # ==============================================================================
    # 6. Plot
    # ==============================================================================
    print("\n" + "=" * 70)
    print("[6] Generating comparison plot...")
    print("=" * 70)

    if len(results) < 2:
        print("ERROR: One or both cases failed. Cannot generate comparison.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        r'Penrose Process: Static Kerr vs. Dynamic Kerr-Vaidya (Escape-Verified)',
        fontsize=15, fontweight='bold', y=0.98
    )

    for ax, label in zip(axes, ['Static', 'Dynamic']):
        res = results[label]
        traj = res['traj']
        cross_idx = res['cross_idx']
        traj2 = res['traj2']
        traj3 = res['traj3']
        m_cross = res['m_cross']
        r_s_cross = res['r_s_cross']
        r_plus_cross = res['r_plus_cross']

        # Ergosphere at split time
        plot_ergosphere_snapshot(ax, m_cross, A, alpha=0.22)

        # For dynamic case, show retreating ergosphere
        if label == 'Dynamic':
            for dt in [30, 70, 120, 180]:
                t_later = t_cross + dt
                m_later = m_func_dynamic(t_later)
                if m_later > M_FLOOR:
                    plot_ergosphere_snapshot(ax, m_later, A, alpha=0.06, color_ergo='#FF4500')

        # Trajectories
        x_pre = traj['r'][:cross_idx+1] * np.cos(traj['phi'][:cross_idx+1])
        y_pre = traj['r'][:cross_idx+1] * np.sin(traj['phi'][:cross_idx+1])
        ax.plot(x_pre, y_pre, color='#1f77b4', linewidth=2.2,
                label=f'Incoming  ($E_1$={E1:.2f})', zorder=3)

        x2 = traj2['r'] * np.cos(traj2['phi'])
        y2 = traj2['r'] * np.sin(traj2['phi'])
        ax.plot(x2, y2, color='#d62728', linewidth=2.0, linestyle='--',
                label=f'Fragment 2  ($E_2$={res["E2"]:.2f}, falls in)', zorder=3)

        x3 = traj3['r'] * np.cos(traj3['phi'])
        y3 = traj3['r'] * np.sin(traj3['phi'])
        ax.plot(x3, y3, color='#2ca02c', linewidth=2.0, linestyle='-.',
                label=f'Fragment 3  ($E_3$={res["E3"]:.2f}, escapes)', zorder=3)

        # Split point
        x_split = traj['r'][cross_idx] * np.cos(traj['phi'][cross_idx])
        y_split = traj['r'][cross_idx] * np.sin(traj['phi'][cross_idx])
        ax.scatter([x_split], [y_split], color='black', s=120, zorder=5,
                   marker='X', edgecolors='white', linewidths=2.0,
                   label='Split point')

        # Start
        ax.scatter([R0 * np.cos(PHI0)], [R0 * np.sin(PHI0)],
                   color='#1f77b4', s=80, zorder=5, marker='o',
                   edgecolors='white', linewidths=1.5)

        # Annotation
        if label == 'Static':
            annotation = (
                f'{label} Kerr\\n'
                f'$a/M = {A:.3f}$\\n'
                f'$m = {m_cross:.3f}M$ (constant)\\n'
                f'$r_s = {r_s_cross:.3f}M$\\n'
                f'$r_+ = {r_plus_cross:.3f}M$\\n'
                f'$p_{{r,3}} = {res["p_r3"]:.2f}$\\n'
                f'$\\eta = {res["eta"]*100:.1f}\\%$'
            )
        else:
            annotation = (
                f'{label} Kerr-Vaidya\\n'
                f'$a/M = {A:.3f}$\\n'
                f'$m(t_{{\\rm split}}) = {m_cross:.3f}M$\\n'
                f'$r_s(t_{{\\rm split}}) = {r_s_cross:.3f}M$\\n'
                f'$r_+ = {r_plus_cross:.3f}M$\\n'
                f'$\\dot{{m}} = -{ALPHA:.4f}$\\n'
                f'$p_{{r,3}} = {res["p_r3"]:.2f}$\\n'
                f'$\\eta = {res["eta"]*100:.1f}\\%$'
            )

        ax.text(0.02, 0.98, annotation,
                transform=ax.transAxes, fontsize=11, va='top', ha='left',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='gray', alpha=0.95))

        ax.set_aspect('equal')
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_xlabel(r'$x \; [M]$', fontsize=12)
        ax.set_ylabel(r'$y \; [M]$', fontsize=12)
        ax.set_title(f'{label} Case', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUT_PDF, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"\nSaved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")

    # ==============================================================================
    # 7. Summary
    # ==============================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Static vs. Dynamic (ESCAPE-VERIFIED)")
    print("=" * 70)

    for label in ['Static', 'Dynamic']:
        res = results[label]
        print(f"\n{label}:")
        print(f"  Mass at split:       m = {res['m_cross']:.4f}M")
        print(f"  Ergosphere at split: r_s = {res['r_s_cross']:.4f}M")
        print(f"  Horizon at split:    r_+ = {res['r_plus_cross']:.4f}M")
        print(f"  Escape momentum:     p_r3 = {res['p_r3']:.2f}")
        print(f"  Fragment 3 final r:  {res['traj3']['r'][-1]:.2f}M")
        print(f"  Efficiency:          eta = {res['eta']*100:.2f}%")

    # Key comparison
    eta_static = results['Static']['eta']
    eta_dynamic = results['Dynamic']['eta']
    delta_eta = (eta_dynamic - eta_static) / eta_static * 100

    print(f"\n{'=' * 70}")
    print(f"KEY RESULT:")
    print(f"  Static efficiency:  {eta_static*100:.2f}%")
    print(f"  Dynamic efficiency: {eta_dynamic*100:.2f}%")
    print(f"  Improvement:        {delta_eta:+.1f}% relative")
    print(f"{'=' * 70}")

    print("\nStep 03 complete.")
    print("=" * 70)