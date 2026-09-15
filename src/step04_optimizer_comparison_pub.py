#!/usr/bin/env python3
"""
================================================================================
Step 04: Optimized Penrose Trajectories — PUBLICATION VERSION
================================================================================

Clean, minimal figure comparing optimal static vs. dynamic Penrose process.
Designed for MNRAS publication.

Run:
    python src/step04_optimizer_comparison_pub.py
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
HORIZON_MIN_DIST = 0.30

R0 = 6.0
PHI0 = 0.0
R_ESCAPE = 5.0

OUT_PDF = os.path.join(project_root, 'figures', 'step04_optimizer_comparison.pdf')
OUT_PNG = os.path.join(project_root, 'figures', 'step04_optimizer_comparison.png')
DPI = 300

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
# Stage 1: Find Skimming Orbit (per metric)
# ==============================================================================

def stage1_find_skimming_orbit(params, e_range=(0.65, 0.95, 14), l_range=(1.4, 3.0, 14), pr_range=(-0.8, -0.05, 12), label=""):
    best = None
    best_score = -1.0
    total = e_range[2] * l_range[2] * pr_range[2]
    count = 0
    E_vals = np.linspace(*e_range)
    L_vals = np.linspace(*l_range)
    p_r0_vals = np.linspace(*pr_range)
    for E in E_vals:
        for L in L_vals:
            for p_r0 in p_r0_vals:
                count += 1
                if count % 500 == 0:
                    print(f"        [{label}] {count}/{total}")
                try:
                    traj = integrate_geodesic(
                        r0=R0, phi0=PHI0, E=E, L=L, p_r0=p_r0,
                        tau_max=250.0, params=params, n_points=1000
                    )
                except Exception:
                    continue
                r = traj['r']
                t = traj['t']
                r_s_vals = 2.0 * params.m_func(t)
                r_plus_vals = params.m_func(t) + np.sqrt(np.maximum(params.m_func(t)**2 - params.a**2, 0.0))
                inside_mask = r <= r_s_vals
                if not np.any(inside_mask):
                    continue
                r_min = np.min(r)
                r_final = r[-1]
                if r_final < r_plus_vals[-1] + 0.05:
                    continue
                if r_final < R0 - 0.5:
                    continue
                inside = np.sum(inside_mask)
                if inside < 30:
                    continue
                horizon_dists = r - r_plus_vals
                min_horizon_dist = np.min(horizon_dists)
                if min_horizon_dist < HORIZON_MIN_DIST:
                    continue
                depth = np.max(r_s_vals) - r_min
                score = depth * inside
                if score > best_score:
                    best_score = score
                    best = {'E': E, 'L': L, 'p_r0': p_r0, 'traj': traj,
                            'r_min': r_min, 'inside_fraction': inside / len(r),
                            'min_horizon_dist': min_horizon_dist}
    return best


def find_deepest_safe_point(traj, m_func, a):
    r_s_vals = 2.0 * m_func(traj['t'])
    r_plus_vals = m_func(traj['t']) + np.sqrt(np.maximum(m_func(traj['t'])**2 - a**2, 0.0))
    inside_mask = traj['r'] <= r_s_vals
    if not np.any(inside_mask):
        return None
    inside_indices = np.where(inside_mask)[0]
    safe_mask = traj['r'][inside_indices] >= r_plus_vals[inside_indices] + HORIZON_MIN_DIST
    if np.any(safe_mask):
        safe_indices = inside_indices[safe_mask]
        return safe_indices[np.argmin(traj['r'][safe_indices])]
    safe_all = traj['r'] >= r_plus_vals + HORIZON_MIN_DIST
    safe_inside = inside_mask & safe_all
    if np.any(safe_inside):
        return np.where(safe_inside)[0][np.argmin(traj['r'][safe_inside])]
    return None


# ==============================================================================
# Stage 2: Brute-Force Search for Escape-Verified Split
# ==============================================================================

def stage2_brute_force(E1, L1, r_split, phi_split, t_split, params, label, a):
    m_split = params.m_func(np.array([t_split]))[0]
    r_plus_split = m_split + np.sqrt(max(m_split**2 - a**2, 0.0))
    gap = r_split - r_plus_split
    if gap < 0.25:
        p_r3_max, n_pr = 15.0, 40
    elif gap < 0.40:
        p_r3_max, n_pr = 10.0, 30
    else:
        p_r3_max, n_pr = 6.0, 25
    E3_vals = np.linspace(E1 + 0.02, E1 + 0.90, 28)
    L3_vals = np.linspace(0.1, L1 - 0.1, 22)
    p_r3_vals = np.linspace(0.3, p_r3_max, n_pr)
    best = None
    best_eta = -1.0
    total = len(E3_vals) * len(L3_vals) * len(p_r3_vals)
    count = 0
    n_valid = 0
    n_escaped = 0
    for E3 in E3_vals:
        for L3 in L3_vals:
            for p_r3 in p_r3_vals:
                count += 1
                if count % 2000 == 0:
                    print(f"        [{label}] {count}/{total} v={n_valid} e={n_escaped}")
                E2 = E1 - E3
                L2 = L1 - L3
                V2 = effective_potential(r_split, m_split, a, E2, L2)
                V3 = effective_potential(r_split, m_split, a, E3, L3)
                if V2 > 0 or V3 > 0:
                    continue
                n_valid += 1
                try:
                    traj3 = integrate_geodesic(
                        r0=r_split, phi0=phi_split, E=E3, L=L3, p_r0=p_r3,
                        tau_max=600.0, params=params, n_points=2000
                    )
                except Exception:
                    continue
                if traj3['r'][-1] > R_ESCAPE:
                    n_escaped += 1
                    eta = penrose_efficiency(E1, E2, E3)
                    if eta > best_eta:
                        best_eta = eta
                        best = {'E2': E2, 'L2': L2, 'E3': E3, 'L3': L3,
                                'p_r3': p_r3, 'eta': eta, 'final_r': traj3['r'][-1]}
    return best


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("Step 04: Optimized Penrose Trajectories (Publication Version)")
    print("=" * 60)

    results = {}

    for label, params in [('Static', PARAMS_STATIC), ('Dynamic', PARAMS_DYNAMIC)]:

        print(f"\n--- {label} ---")

        candidate = stage1_find_skimming_orbit(params, label=label)
        if candidate is None:
            print(f"ERROR: No skimming orbit for {label}")
            continue

        E1 = candidate['E']
        L1 = candidate['L']
        p_r0 = candidate['p_r0']
        print(f"  E1={E1:.4f} L1={L1:.4f} p_r0={p_r0:.4f} r_min={candidate['r_min']:.4f}")

        traj_in = integrate_geodesic(
            r0=R0, phi0=PHI0, E=E1, L=L1, p_r0=p_r0,
            tau_max=300.0, params=params, n_points=1500
        )

        deepest_idx = find_deepest_safe_point(traj_in, params.m_func, A)
        if deepest_idx is None:
            print(f"  ERROR: No safe split point")
            continue

        t_split = traj_in['t'][deepest_idx]
        r_split = traj_in['r'][deepest_idx]
        phi_split = traj_in['phi'][deepest_idx]
        m_split = params.m_func(np.array([t_split]))[0]
        r_plus_split = m_split + np.sqrt(max(m_split**2 - A**2, 0.0))
        print(f"  Split: t={t_split:.2f} r={r_split:.4f}")

        opt = stage2_brute_force(E1, L1, r_split, phi_split, t_split, params, label, A)
        if opt is None:
            print(f"  ERROR: No escape-verified split")
            continue

        traj2 = integrate_geodesic(
            r0=r_split, phi0=phi_split, E=opt['E2'], L=opt['L2'], p_r0=-0.05,
            tau_max=100.0, params=params, n_points=800
        )
        traj3 = integrate_geodesic(
            r0=r_split, phi0=phi_split, E=opt['E3'], L=opt['L3'], p_r0=opt['p_r3'],
            tau_max=600.0, params=params, n_points=2500
        )

        verified = bool(traj3['r'][-1] > R_ESCAPE)
        print(f"  eta={opt['eta']*100:.2f}% verified={verified}")

        results[label] = {
            'traj_in': traj_in, 'deepest_idx': deepest_idx,
            'traj2': traj2, 'traj3': traj3, 'opt': opt,
            'E1': E1, 'L1': L1, 'm_split': m_split,
            'r_plus': r_plus_split, 'r_s': 2.0 * m_split,
            'verified': verified
        }

    # ==============================================================================
    # Plot — clean, minimal, publication-ready
    # ==============================================================================
    print("\nGenerating figure...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, label in zip(axes, ['Static', 'Dynamic']):
        if label not in results:
            continue
        res = results[label]
        traj = res['traj_in']
        idx = res['deepest_idx']
        traj2 = res['traj2']
        traj3 = res['traj3']
        opt = res['opt']
        m_s = res['m_split']

        # Geometry
        plot_ergosphere_and_horizon(m=m_s, a=A, ax=ax, label=False)

        # For dynamic, show one earlier ergosphere
        if label == 'Dynamic':
            m_early = max(M_FLOOR, m_s + 0.03)
            if m_early > M_FLOOR:
                theta = np.linspace(0, 2*np.pi, 200)
                r_s_early = 2.0 * m_early
                ax.plot(r_s_early*np.cos(theta), r_s_early*np.sin(theta),
                        '--', color='#ff7f0e', linewidth=1.2, alpha=0.6)

        # Trajectories
        x_pre = traj['r'][:idx+1] * np.cos(traj['phi'][:idx+1])
        y_pre = traj['r'][:idx+1] * np.sin(traj['phi'][:idx+1])
        x2 = traj2['r'] * np.cos(traj2['phi'])
        y2 = traj2['r'] * np.sin(traj2['phi'])
        x3 = traj3['r'] * np.cos(traj3['phi'])
        y3 = traj3['r'] * np.sin(traj3['phi'])

        ax.plot(x_pre, y_pre, color='#1f77b4', linewidth=2.2, label='Incoming', zorder=3)
        ax.plot(x2, y2, color='#d62728', linewidth=1.8, linestyle='--', label='Fragment 2 (plunges)', zorder=3)
        ax.plot(x3, y3, color='#2ca02c', linewidth=1.8, linestyle='-.', label='Fragment 3 (escapes)', zorder=3)

        # Split point
        xs = traj['r'][idx] * np.cos(traj['phi'][idx])
        ys = traj['r'][idx] * np.sin(traj['phi'][idx])
        ax.scatter([xs], [ys], color='black', s=100, zorder=5, marker='X',
                   edgecolors='white', linewidths=1.5)

        # Start point
        ax.scatter([R0*np.cos(PHI0)], [R0*np.sin(PHI0)], color='#1f77b4',
                   s=50, zorder=5, marker='o', edgecolors='white', linewidths=1)

        # Tight limits
        all_x = np.concatenate([x_pre, x2, x3])
        all_y = np.concatenate([y_pre, y2, y3])
        margin = 0.5
        max_r = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) + margin
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)

        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \; [M]$', fontsize=11)
        ax.set_ylabel(r'$y \; [M]$', fontsize=11)

        # Minimal annotation
        if label == 'Static':
            title = 'Static Kerr'
            ann = rf'$\eta = {opt["eta"]*100:.1f}\%$' + '\n' + rf'$m = {m_s:.3f}M$'
        else:
            title = 'Dynamic Kerr--Vaidya'
            ann = rf'$\eta = {opt["eta"]*100:.1f}\%$' + '\n' + rf'$m = {m_s:.3f}M$' + '\n' + rf'$\dot{{m}} = -{ALPHA:.4f}$'

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=10,
                va='top', ha='left', linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='lightgray', alpha=0.9))

        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUT_PDF, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
    print("Done.")