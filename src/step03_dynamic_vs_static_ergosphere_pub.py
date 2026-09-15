#!/usr/bin/env python3
"""
================================================================================
Step 03: Static vs. Dynamic Ergosphere — PUBLICATION VERSION
================================================================================

Clean, minimal figure showing the Penrose process in static Kerr vs.
shrinking-ergosphere Kerr-Vaidya. Designed for MNRAS publication.

Run:
    python src/step03_dynamic_vs_static_ergosphere_pub.py
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

warnings.filterwarnings('ignore')

from kerr_vaidya_metric import (
    KerrVaidyaParams,
    integrate_geodesic,
    effective_potential,
    penrose_efficiency,
    plot_ergosphere_and_horizon,
)

# ==============================================================================
# Configuration
# ==============================================================================

M0 = 1.0
A = 0.9
ALPHA = 0.0005
M_FLOOR = 0.95

R0 = 6.0
PHI0 = 0.0
R_ESCAPE = 5.0

OUT_PNG = os.path.join(project_root, 'figures', 'step03_dynamic_vs_static_ergosphere.pdf')
DPI = 300

# Shared initial conditions (from Step 04 fair comparison attempt)
E_SHARED = 0.95
L_SHARED = 2.05
P_R0_SHARED = -0.45

# Split parameters (from successful Step 04 runs)
E3_STATIC = 1.35
L3_STATIC = 1.85
P_R3_STATIC = 5.77

E3_DYNAMIC = 1.25
L3_DYNAMIC = 1.85
P_R3_DYNAMIC = 5.77

# ==============================================================================
# Metric factories
# ==============================================================================

def m_func_static(t):
    t = np.asarray(t)
    return np.full_like(t, M0, dtype=float)

def m_dot_func_static(t):
    t = np.asarray(t)
    return np.zeros_like(t, dtype=float)

def m_func_dynamic(t):
    t = np.asarray(t)
    return np.maximum(M_FLOOR, M0 - ALPHA * t)

def m_dot_func_dynamic(t):
    t = np.asarray(t)
    m = M0 - ALPHA * t
    return np.where(m > M_FLOOR, -ALPHA, 0.0)

PARAMS_STATIC = KerrVaidyaParams(a=A, m_func=m_func_static, m_dot_func=m_dot_func_static)
PARAMS_DYNAMIC = KerrVaidyaParams(a=A, m_func=m_func_dynamic, m_dot_func=m_dot_func_dynamic)

# ==============================================================================
# Integration
# ==============================================================================

def run_case(params, label):
    print(f"[{label}] Integrating incoming orbit...")
    traj_in = integrate_geodesic(
        r0=R0, phi0=PHI0, E=E_SHARED, L=L_SHARED, p_r0=P_R0_SHARED,
        tau_max=150.0, params=params, n_points=800
    )

    # Find deepest point inside ergosphere
    r_s_vals = 2.0 * params.m_func(traj_in['t'])
    inside_mask = traj_in['r'] <= r_s_vals
    if not np.any(inside_mask):
        print(f"[{label}] WARNING: No point inside ergosphere!")
        return None

    inside_indices = np.where(inside_mask)[0]
    deepest_idx = inside_indices[np.argmin(traj_in['r'][inside_indices])]

    t_split = traj_in['t'][deepest_idx]
    r_split = traj_in['r'][deepest_idx]
    phi_split = traj_in['phi'][deepest_idx]
    m_split = params.m_func(np.array([t_split]))[0]

    E2 = E_SHARED - E3_STATIC if label == 'Static' else E_SHARED - E3_DYNAMIC
    L2 = L_SHARED - L3_STATIC if label == 'Static' else L_SHARED - E3_DYNAMIC
    E3 = E3_STATIC if label == 'Static' else E3_DYNAMIC
    L3 = L3_STATIC if label == 'Static' else L3_DYNAMIC
    p_r3 = P_R3_STATIC if label == 'Static' else P_R3_DYNAMIC

    print(f"[{label}] Split at t={t_split:.2f}M, r={r_split:.4f}M")

    traj2 = integrate_geodesic(
        r0=r_split, phi0=phi_split, E=E2, L=L2, p_r0=-0.1,
        tau_max=50.0, params=params, n_points=400
    )

    traj3 = integrate_geodesic(
        r0=r_split, phi0=phi_split, E=E3, L=L3, p_r0=p_r3,
        tau_max=200.0, params=params, n_points=1000
    )

    eta = penrose_efficiency(E_SHARED, E2, E3)
    verified = traj3['r'][-1] > R_ESCAPE
    print(f"[{label}] eta={eta*100:.1f}%, verified_escape={verified}")

    return {
        'traj_in': traj_in, 'deepest_idx': deepest_idx,
        'traj2': traj2, 'traj3': traj3,
        't_split': t_split, 'r_split': r_split, 'phi_split': phi_split,
        'm_split': m_split, 'eta': eta, 'verified': verified,
        'E2': E2, 'L2': L2, 'E3': E3, 'L3': L3,
    }


# ==============================================================================
# Plot — clean, minimal, publication-ready
# ==============================================================================

def plot_clean(results_static, results_dynamic):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, label, res in zip(axes, ['Static', 'Dynamic'], [results_static, results_dynamic]):
        if res is None:
            continue

        traj = res['traj_in']
        idx = res['deepest_idx']
        traj2 = res['traj2']
        traj3 = res['traj3']
        m_s = res['m_split']

        # Geometry
        plot_ergosphere_and_horizon(m=m_s, a=A, ax=ax, label=False)

        # For dynamic, show ONE earlier ergosphere to illustrate shrinkage
        if label == 'Dynamic':
            m_early = max(M_FLOOR, m_s + 0.05)  # slightly larger = earlier time
            if m_early > M_FLOOR:
                theta = np.linspace(0, 2*np.pi, 200)
                r_s_early = 2.0 * m_early
                ax.plot(r_s_early*np.cos(theta), r_s_early*np.sin(theta),
                        '--', color='#ff7f0e', linewidth=1.2, alpha=0.6)

        # Trajectories
        x_in = traj['r'] * np.cos(traj['phi'])
        y_in = traj['r'] * np.sin(traj['phi'])
        x_pre = traj['r'][:idx+1] * np.cos(traj['phi'][:idx+1])
        y_pre = traj['r'][:idx+1] * np.sin(traj['phi'][:idx+1])
        x2 = traj2['r'] * np.cos(traj2['phi'])
        y2 = traj2['r'] * np.sin(traj2['phi'])
        x3 = traj3['r'] * np.cos(traj3['phi'])
        y3 = traj3['r'] * np.sin(traj3['phi'])

        ax.plot(x_pre, y_pre, color='#1f77b4', linewidth=2.0, label='Incoming', zorder=3)
        ax.plot(x2, y2, color='#d62728', linewidth=1.8, linestyle='--', label='Fragment 2 (plunges)', zorder=3)
        ax.plot(x3, y3, color='#2ca02c', linewidth=1.8, linestyle='-.', label='Fragment 3 (escapes)', zorder=3)

        # Split point
        xs = traj['r'][idx] * np.cos(traj['phi'][idx])
        ys = traj['r'][idx] * np.sin(traj['phi'][idx])
        ax.scatter([xs], [ys], color='black', s=100, zorder=5, marker='X',
                   edgecolors='white', linewidths=1.5)

        # Start point
        ax.scatter([R0*np.cos(PHI0)], [R0*np.sin(PHI0)], color='#1f77b4',
                   s=60, zorder=5, marker='o', edgecolors='white', linewidths=1)

        # Tight limits
        all_x = np.concatenate([x_in, x2, x3])
        all_y = np.concatenate([y_in, y2, y3])
        margin = 0.5
        max_r = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) + margin
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)

        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \; [M]$', fontsize=11)
        ax.set_ylabel(r'$y \; [M]$', fontsize=11)

        # Minimal annotation — just efficiency and mass
        if label == 'Static':
            title = 'Static Kerr'
            ann = rf'$\eta = {res["eta"]*100:.1f}\%$' + '\n' + rf'$m = {m_s:.3f}M$'
        else:
            title = 'Dynamic Kerr--Vaidya'
            ann = rf'$\eta = {res["eta"]*100:.1f}\%$' + '\n' + rf'$m = {m_s:.3f}M$' + '\n' + rf'$\dot{{m}} = -{ALPHA:.4f}$'

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=10,
                va='top', ha='left', linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='lightgray', alpha=0.9))

        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\nSaved: {OUT_PNG}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Step 03: Static vs. Dynamic Ergosphere (Publication Version)")
    print("=" * 60)

    res_s = run_case(PARAMS_STATIC, 'Static')
    res_d = run_case(PARAMS_DYNAMIC, 'Dynamic')

    if res_s and res_d:
        plot_clean(res_s, res_d)
        print("\nDone.")
    else:
        print("\nERROR: One or both cases failed.")