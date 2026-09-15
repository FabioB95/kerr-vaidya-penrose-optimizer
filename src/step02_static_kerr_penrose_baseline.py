#!/usr/bin/env python3
"""
================================================================================
FILE: src/step02_static_kerr_penrose_baseline.py
================================================================================

Step 02: Baseline Static Kerr Penrose Process — Fast Version
============================================================

Uses a coarse parameter search (125 combinations) with suppressed warnings
for speed. If no plunging orbit is found, falls back to pre-computed
initial conditions that are known to work for a/M = 0.9.

Outputs:
    figures/step02_penrose_trajectory.png
    figures/step02_penrose_trajectory.pdf

Run:
    python src/step02_static_kerr_penrose_baseline.py
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
    plot_ergosphere_and_horizon,
    event_horizon_radius,
    ergosphere_boundary_equatorial,
    effective_potential,
    penrose_efficiency,
    max_penrose_efficiency_static_kerr,
)

# Suppress numpy/scipy warnings during the brute-force search
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

M = 1.0
A = 0.9
PARAMS = KerrVaidyaParams(a=A)

R0 = 6.0
PHI0 = 0.0
TAU_MAX = 500.0
N_POINTS = 2000

OUT_PNG = os.path.join(project_root, 'figures', 'step02_penrose_trajectory.png')
OUT_PDF = os.path.join(project_root, 'figures', 'step02_penrose_trajectory.pdf')
DPI = 300

# Pre-computed fallback initial conditions for a/M = 0.9
# These are known to produce a plunging orbit into the ergosphere
FALLBACK = {
    'E': 0.85,
    'L': 1.20,
    'p_r0': -0.60
}


# ==============================================================================
# Helper: Fast search for plunging orbit
# ==============================================================================

def search_plunging_orbit(r0, phi0, m, a, params,
                          e_range=(0.75, 0.95, 5),
                          l_range=(1.0, 2.0, 5),
                          pr_range=(-0.8, -0.2, 5)):
    """
    Fast coarse search. Only 125 combinations.
    """
    r_s = ergosphere_boundary_equatorial(m, a)

    best = None
    best_depth = 0.0

    E_vals = np.linspace(*e_range)
    L_vals = np.linspace(*l_range)
    Pr_vals = np.linspace(*pr_range)

    total = len(E_vals) * len(L_vals) * len(Pr_vals)
    print(f"    Fast search: {total} combinations...")

    for E in E_vals:
        for L in L_vals:
            for p_r0 in Pr_vals:
                try:
                    traj = integrate_geodesic(
                        r0=r0, phi0=phi0,
                        E=E, L=L, p_r0=p_r0,
                        tau_max=200.0,
                        params=params,
                        n_points=200
                    )
                except Exception:
                    continue

                # Check if any point is inside the ergosphere
                inside = traj['r'] <= r_s
                if np.any(inside):
                    r_min = np.min(traj['r'])
                    depth = r_s - r_min
                    if depth > best_depth:
                        best_depth = depth
                        best = {
                            'E': E, 'L': L, 'p_r0': p_r0,
                            'r_min': r_min, 'traj': traj,
                            'depth': depth
                        }

    return best


def find_ergosphere_crossing(traj, m, a):
    r_s = ergosphere_boundary_equatorial(m, a)
    for i, r in enumerate(traj['r']):
        if r <= r_s:
            return i
    return None


def find_valid_split(r_split, m, a, E1, L1, n_samples=60):
    best = None
    best_eta = -1.0

    E2_vals = np.linspace(-0.35, -0.02, n_samples)
    L2_vals = np.linspace(0.1, L1 - 0.1, n_samples)

    for E2 in E2_vals:
        E3 = E1 - E2
        if E3 <= 0:
            continue
        for L2 in L2_vals:
            L3 = L1 - L2
            if L3 <= 0:
                continue

            V2 = effective_potential(r_split, m, a, E2, L2)
            V3 = effective_potential(r_split, m, a, E3, L3)

            if V2 <= 0 and V3 <= 0:
                eta = penrose_efficiency(E1, E2, E3)
                if eta > best_eta:
                    best_eta = eta
                    best = {
                        'E2': E2, 'L2': L2,
                        'E3': E3, 'L3': L3,
                        'eta': eta,
                        'V2': V2, 'V3': V3
                    }

    return best


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("Step 02: Static Kerr Penrose Process Baseline")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Search for plunging orbit
    # -------------------------------------------------------------------------
    print("\n[1] Searching for plunging orbit...")

    candidate = search_plunging_orbit(R0, PHI0, M, A, PARAMS)

    if candidate is None:
        print("    No plunging orbit found in coarse search.")
        print("    Using fallback initial conditions...")
        candidate = {
            'E': FALLBACK['E'],
            'L': FALLBACK['L'],
            'p_r0': FALLBACK['p_r0'],
            'traj': None
        }

    E1 = candidate['E']
    L1 = candidate['L']
    p_r0 = candidate['p_r0']

    print(f"    Selected: E={E1:.4f}, L={L1:.4f}, p_r0={p_r0:.4f}")
    if candidate.get('r_min'):
        print(f"    Coarse search r_min = {candidate['r_min']:.4f}M")

    # High-resolution re-integration
    print("\n    Re-integrating with high resolution...")
    traj = integrate_geodesic(
        r0=R0, phi0=PHI0,
        E=E1, L=L1, p_r0=p_r0,
        tau_max=TAU_MAX,
        params=PARAMS,
        n_points=N_POINTS
    )

    print(f"    Status: {traj['message']}")
    print(f"    Final radius: r = {traj['r'][-1]:.4f}M")

    # -------------------------------------------------------------------------
    # 2. Find ergosphere crossing
    # -------------------------------------------------------------------------
    print("\n[2] Locating ergosphere entry...")

    r_plus, _ = event_horizon_radius(M, A)
    r_s = ergosphere_boundary_equatorial(M, A)
    print(f"    Horizon: r_+ = {r_plus:.4f}M")
    print(f"    Ergosphere: r_s = {r_s:.4f}M")

    cross_idx = find_ergosphere_crossing(traj, M, A)
    if cross_idx is None:
        print("    ERROR: Trajectory did not enter ergosphere!")
        sys.exit(1)

    print(f"    Entered at tau = {traj['tau'][cross_idx]:.2f}")
    print(f"    Entry: r={traj['r'][cross_idx]:.4f}M, phi={traj['phi'][cross_idx]:.4f}")

    # -------------------------------------------------------------------------
    # 3. Find optimal Penrose split
    # -------------------------------------------------------------------------
    print("\n[3] Searching for optimal Penrose split...")

    r_split = traj['r'][cross_idx]
    phi_split = traj['phi'][cross_idx]

    split = find_valid_split(r_split, M, A, E1, L1, n_samples=60)

    if split is None:
        print("    ERROR: No valid split found!")
        sys.exit(1)

    E2 = split['E2']
    L2 = split['L2']
    E3 = split['E3']
    L3 = split['L3']
    eta = split['eta']

    print(f"    Optimal split:")
    print(f"    Fragment 2 (infalling): E2={E2:.4f}, L2={L2:.4f}, V_eff={split['V2']:.4f}")
    print(f"    Fragment 3 (escaping):  E3={E3:.4f}, L3={L3:.4f}, V_eff={split['V3']:.4f}")

    # -------------------------------------------------------------------------
    # 4. Integrate post-split trajectories
    # -------------------------------------------------------------------------
    print("\n[4] Integrating post-split trajectories...")

    traj2 = integrate_geodesic(
        r0=r_split, phi0=phi_split,
        E=E2, L=L2, p_r0=-0.1,
        tau_max=100.0,
        params=PARAMS,
        n_points=500
    )

    # Try increasing p_r0 until fragment 3 escapes
    p_r3 = 0.2
    for _ in range(10):
        traj3 = integrate_geodesic(
            r0=r_split, phi0=phi_split,
            E=E3, L=L3, p_r0=p_r3,
            tau_max=400.0,
            params=PARAMS,
            n_points=1000
        )
        if traj3['r'][-1] > 5.0:  # Escaped beyond 5M
            break
        p_r3 += 0.3

    

    print(f"    Fragment 2: final r = {traj2['r'][-1]:.4f}M")
    print(f"    Fragment 3: final r = {traj3['r'][-1]:.4f}M")

    # -------------------------------------------------------------------------
    # 5. Efficiency
    # -------------------------------------------------------------------------
    eta_theory = max_penrose_efficiency_static_kerr(M, A)

    print("\n[5] Energy extraction:")
    print(f"    Input:     E1 = {E1:.4f}")
    print(f"    Infalling: E2 = {E2:.4f}  (negative!)")
    print(f"    Escaping:  E3 = {E3:.4f}")
    print(f"    Efficiency: eta = {eta*100:.2f}%")
    print(f"    Theoretical max (approx): eta_max = {eta_theory*100:.2f}%")

    # -------------------------------------------------------------------------
    # 6. Plot
    # -------------------------------------------------------------------------
    print("\n[6] Generating plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    plot_ergosphere_and_horizon(m=M, a=A, ax=ax, label=False)

    # Pre-split
    x_pre = traj['r'][:cross_idx+1] * np.cos(traj['phi'][:cross_idx+1])
    y_pre = traj['r'][:cross_idx+1] * np.sin(traj['phi'][:cross_idx+1])
    ax.plot(x_pre, y_pre, color='#1f77b4', linewidth=2.0,
            label=f'Incoming  (E={E1:.2f})', zorder=3)

    # Fragment 2
    x2 = traj2['r'] * np.cos(traj2['phi'])
    y2 = traj2['r'] * np.sin(traj2['phi'])
    ax.plot(x2, y2, color='#d62728', linewidth=2.0, linestyle='--',
            label=f'Fragment 2  (E={E2:.2f}, falls in)', zorder=3)

    # Fragment 3
    x3 = traj3['r'] * np.cos(traj3['phi'])
    y3 = traj3['r'] * np.sin(traj3['phi'])
    ax.plot(x3, y3, color='#2ca02c', linewidth=2.0, linestyle='-.',
            label=f'Fragment 3  (E={E3:.2f}, escapes)', zorder=3)

    # Split point
    x_split = r_split * np.cos(phi_split)
    y_split = r_split * np.sin(phi_split)
    ax.scatter([x_split], [y_split], color='black', s=120, zorder=5,
               marker='X', edgecolors='white', linewidths=2.0,
               label='Split point')

    # Start
    ax.scatter([R0 * np.cos(PHI0)], [R0 * np.sin(PHI0)],
               color='#1f77b4', s=80, zorder=5, marker='o',
               edgecolors='white', linewidths=1.5)

    # Annotation
    ax.text(0.02, 0.98,
            f'$a/M = {A:.3f}$\\n'
            f'$r_+ = {r_plus:.3f}M$\\n'
            f'$r_s = {r_s:.3f}M$\\n'
            f'$E_1 = {E1:.3f}$\\n'
            f'$E_2 = {E2:.3f}$\\n'
            f'$E_3 = {E3:.3f}$\\n'
            f'$\\eta = {eta*100:.1f}\\%$',
            transform=ax.transAxes, fontsize=11, va='top', ha='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.95))

    ax.set_aspect('equal')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel(r'$x \; [M]$', fontsize=12)
    ax.set_ylabel(r'$y \; [M]$', fontsize=12)
    ax.set_title(r'Static Kerr: Penrose Process Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUT_PDF, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"    Saved: {OUT_PNG}")
    print(f"    Saved: {OUT_PDF}")
    print("\n" + "=" * 60)
    print("Step 02 complete.")
    print("=" * 60)