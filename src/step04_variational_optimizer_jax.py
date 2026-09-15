#!/usr/bin/env python3
"""
================================================================================
FILE: src/step04_variational_optimizer_skimming.py
================================================================================

Step 04: Skimming-Orbit Variational Optimizer
=============================================

Key insight: The Penrose process requires the particle to SKIM the ergosphere,
not plunge through it. We search for initial conditions where the particle
enters the ergosphere, reaches a minimum radius r_min with r_+ < r_min < r_s,
and survives for a significant time without hitting the horizon.

Then we split at the point of deepest penetration into the ergosphere and
search for (E3, L3, p_r3) that allows fragment 3 to escape.

Outputs:
    figures/step04_optimizer_comparison.png
    figures/step04_optimizer_comparison.pdf

Run:
    python src/step04_variational_optimizer_skimming.py
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
    event_horizon_radius,
    effective_potential,
    penrose_efficiency,
    plot_ergosphere_and_horizon,
)

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

# FIX: Use np.maximum instead of max for array compatibility
def m_func_dynamic(t):
    return np.maximum(M_FLOOR, M0 - ALPHA * t)

def m_dot_func_dynamic(t):
    m = M0 - ALPHA * t
    return -ALPHA if m > M_FLOOR else 0.0

PARAMS_STATIC = KerrVaidyaParams(a=A, m_func=m_func_static, m_dot_func=m_dot_func_static)
PARAMS_DYNAMIC = KerrVaidyaParams(a=A, m_func=m_func_dynamic, m_dot_func=m_dot_func_dynamic)

R0 = 6.0
PHI0 = 0.0
R_ESCAPE = 5.0

OUT_PNG = os.path.join(project_root, 'figures', 'step04_optimizer_comparison.png')
OUT_PDF = os.path.join(project_root, 'figures', 'step04_optimizer_comparison.pdf')
DPI = 300

# ==============================================================================
# Stage 1: Find Skimming Orbit
# ==============================================================================

def stage1_find_skimming_orbit(params, e_range=(0.78, 0.95, 15), l_range=(1.8, 2.5, 15), pr_range=(-0.5, -0.05, 10)):
    """
    Search for orbits that:
    1. Enter the ergosphere (r < r_s)
    2. Do NOT hit the horizon within tau=150
    3. Reach a minimum radius r_min with r_+ < r_min < r_s (inside ergosphere but not at horizon)
    
    These are "skimming" orbits that spend time in the ergosphere.
    """
    best = None
    best_score = -1.0
    
    r_s = 2.0 * M0
    r_plus = M0 + np.sqrt(max(M0**2 - A**2, 0.0))
    
    total = e_range[2] * l_range[2] * pr_range[2]
    count = 0
    
    for E in np.linspace(*e_range):
        for L in np.linspace(*l_range):
            for p_r0 in np.linspace(*pr_range):
                count += 1
                if count % 500 == 0:
                    print(f"        ...{count}/{total} tested")
                
                try:
                    traj = integrate_geodesic(
                        r0=R0, phi0=PHI0, E=E, L=L, p_r0=p_r0,
                        tau_max=150.0, params=params, n_points=500
                    )
                except Exception:
                    continue
                
                r_min = np.min(traj['r'])
                r_final = traj['r'][-1]
                
                # Must enter ergosphere
                if r_min > r_s:
                    continue
                
                # Must NOT hit horizon (final r should be > r_+ + 0.1)
                if r_final < r_plus + 0.1:
                    continue
                
                # Must spend time inside ergosphere (at least 20% of trajectory)
                inside = np.sum(traj['r'] <= r_s)
                if inside < 50:
                    continue
                
                # Score: deeper penetration = better (more negative energy possible)
                # But not too deep (r_min close to r_+ is dangerous)
                # Ideal: r_min around 1.6-1.8M for a=0.9
                depth = r_s - r_min
                score = depth * inside  # Deeper and longer = better
                
                if score > best_score:
                    best_score = score
                    best = {
                        'E': E, 'L': L, 'p_r0': p_r0,
                        'traj': traj, 'r_min': r_min,
                        'inside_fraction': inside / len(traj['r'])
                    }
    
    return best


def find_deepest_point(traj, m_func):
    """Find the index of deepest penetration into the ergosphere."""
    # FIX: m_func now handles arrays via np.maximum
    r_s_vals = 2.0 * m_func(traj['t'])
    inside_mask = traj['r'] <= r_s_vals
    if not np.any(inside_mask):
        return None
    inside_indices = np.where(inside_mask)[0]
    deepest_idx = inside_indices[np.argmin(traj['r'][inside_indices])]
    return deepest_idx


# ==============================================================================
# Stage 2: Brute-Force Search for Escape-Verified Split
# ==============================================================================

def stage2_brute_force(E1, L1, r_split, phi_split, t_split, params, label):
    """
    Search over (E3, L3, p_r3) with expanded ranges.
    """
    print(f"\n    [{label}] Stage 2: Brute-force escape-verified search...")
    
    m_split = params.m_func(t_split)
    
    # EXPANDED search ranges
    E3_vals = np.linspace(E1 + 0.05, E1 + 0.80, 25)
    L3_vals = np.linspace(0.2, L1 - 0.2, 20)
    p_r3_vals = np.linspace(0.5, 6.0, 25)
    
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
                    print(f"        ...{count}/{total} tested, {n_valid} valid, {n_escaped} escaped")
                
                E2 = E1 - E3
                L2 = L1 - L3
                
                V2 = effective_potential(r_split, m_split, A, E2, L2)
                V3 = effective_potential(r_split, m_split, A, E3, L3)
                if V2 > 0 or V3 > 0:
                    continue
                
                n_valid += 1
                
                try:
                    traj3 = integrate_geodesic(
                        r0=r_split, phi0=phi_split,
                        E=E3, L=L3, p_r0=p_r3,
                        tau_max=400.0, params=params, n_points=800
                    )
                except Exception:
                    continue
                
                final_r = traj3['r'][-1]
                if final_r > R_ESCAPE:
                    n_escaped += 1
                    eta = penrose_efficiency(E1, E2, E3)
                    if eta > best_eta:
                        best_eta = eta
                        best = {
                            'E2': E2, 'L2': L2,
                            'E3': E3, 'L3': L3,
                            'p_r3': p_r3,
                            'eta': eta,
                            'final_r': final_r
                        }
    
    print(f"    [{label}] Complete: {n_valid} valid V_eff, {n_escaped} escaped")
    
    if best is None:
        print(f"    [{label}] WARNING: No escape-verified split found!")
        return None
    
    print(f"    [{label}] BEST: E3={best['E3']:.4f}, L3={best['L3']:.4f}, p_r3={best['p_r3']:.4f}")
    print(f"    [{label}] eta={best['eta']*100:.2f}%, final_r={best['final_r']:.2f}M")
    
    return best


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':

    print("=" * 70)
    print("Step 04: Skimming-Orbit Variational Optimizer")
    print("=" * 70)

    results = {}

    for label, params in [('Static', PARAMS_STATIC), ('Dynamic', PARAMS_DYNAMIC)]:

        print(f"\n{'-' * 60}")
        print(f"Case: {label}")
        print(f"{'-' * 60}")

        # ---------------------------------------------------------------------
        # 1. Find skimming orbit
        # ---------------------------------------------------------------------
        print(f"\n[1] Stage 1: Searching for skimming orbit ({label})...")
        print(f"    (Particle must enter ergosphere but NOT hit horizon quickly)")
        
        candidate = stage1_find_skimming_orbit(params)

        if candidate is None:
            print(f"    ERROR: No skimming orbit found!")
            continue

        E1 = candidate['E']
        L1 = candidate['L']
        p_r0 = candidate['p_r0']
        print(f"    Selected: E={E1:.4f}, L={L1:.4f}, p_r0={p_r0:.4f}")
        print(f"    Minimum radius: r_min={candidate['r_min']:.4f}M")
        print(f"    Time inside ergosphere: {candidate['inside_fraction']*100:.1f}%")

        traj_in = candidate['traj']

        # ---------------------------------------------------------------------
        # 2. Find deepest point for split
        # ---------------------------------------------------------------------
        print(f"\n[2] Finding optimal split point ({label})...")
        
        deepest_idx = find_deepest_point(traj_in, params.m_func)
        if deepest_idx is None:
            print(f"    ERROR: No point inside ergosphere found!")
            continue

        t_split = traj_in['t'][deepest_idx]
        r_split = traj_in['r'][deepest_idx]
        phi_split = traj_in['phi'][deepest_idx]
        m_split = params.m_func(t_split)

        print(f"    Split at deepest point:")
        print(f"    t={t_split:.2f}M, r={r_split:.4f}M, phi={phi_split:.4f}")
        print(f"    m={m_split:.4f}M, r_s={2*m_split:.4f}M")

        # ---------------------------------------------------------------------
        # 3. Brute-force search for escape-verified split
        # ---------------------------------------------------------------------
        print(f"\n[3] Stage 2: Brute-force search ({label})...")

        opt_result = stage2_brute_force(
            E1, L1, r_split, phi_split, t_split, params, label
        )

        if opt_result is None:
            print(f"    ERROR: No valid split found for {label}!")
            continue

        # ---------------------------------------------------------------------
        # 4. Final high-res integration
        # ---------------------------------------------------------------------
        print(f"\n[4] Final integration ({label})...")

        traj2 = integrate_geodesic(
            r0=r_split, phi0=phi_split,
            E=opt_result['E2'], L=opt_result['L2'], p_r0=-0.1,
            tau_max=100.0, params=params, n_points=500
        )

        traj3 = integrate_geodesic(
            r0=r_split, phi0=phi_split,
            E=opt_result['E3'], L=opt_result['L3'], p_r0=opt_result['p_r3'],
            tau_max=400.0, params=params, n_points=2000
        )

        print(f"    Fragment 2 final r: {traj2['r'][-1]:.3f}M")
        print(f"    Fragment 3 final r: {traj3['r'][-1]:.3f}M")

        verified = traj3['r'][-1] > R_ESCAPE
        print(f"    VERIFIED ESCAPE: {'YES' if verified else 'NO'}")

        r_plus = m_split + np.sqrt(max(m_split**2 - A**2, 0.0))
        r_s = 2.0 * m_split

        results[label] = {
            'traj_in': traj_in, 'deepest_idx': deepest_idx,
            'traj2': traj2, 'traj3': traj3,
            'opt': opt_result,
            'm_split': m_split, 'r_plus': r_plus, 'r_s': r_s,
            'E1': E1, 'L1': L1, 'verified': verified
        }

    # ==============================================================================
    # 5. Plot
    # ==============================================================================
    print("\n" + "=" * 70)
    print("[5] Generating comparison plot...")
    print("=" * 70)

    if len(results) < 2:
        print("ERROR: One or both cases failed.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        r'Skimming-Orbit Optimization: Static vs. Dynamic Kerr-Vaidya Penrose Process',
        fontsize=15, fontweight='bold', y=0.98
    )

    for ax, label in zip(axes, ['Static', 'Dynamic']):
        res = results[label]
        traj = res['traj_in']
        deepest_idx = res['deepest_idx']
        traj2 = res['traj2']
        traj3 = res['traj3']
        opt = res['opt']

        plot_ergosphere_and_horizon(m=res['m_split'], a=A, ax=ax, label=False)

        if label == 'Dynamic':
            for dt in [20, 50, 100]:
                t_later = traj['t'][deepest_idx] + dt
                m_later = m_func_dynamic(t_later)
                if m_later > M_FLOOR:
                    plot_ergosphere_and_horizon(m=m_later, a=A, ax=ax, label=False)

        # Full incoming trajectory (including inside ergosphere)
        x_in = traj['r'] * np.cos(traj['phi'])
        y_in = traj['r'] * np.sin(traj['phi'])
        ax.plot(x_in, y_in, color='#1f77b4', linewidth=1.5, alpha=0.5,
                label=f'Full incoming', zorder=2)

        # Pre-split trajectory (up to deepest point)
        x_pre = traj['r'][:deepest_idx+1] * np.cos(traj['phi'][:deepest_idx+1])
        y_pre = traj['r'][:deepest_idx+1] * np.sin(traj['phi'][:deepest_idx+1])
        ax.plot(x_pre, y_pre, color='#1f77b4', linewidth=2.5,
                label=f'Incoming  ($E_1$={res["E1"]:.2f})', zorder=3)

        x2 = traj2['r'] * np.cos(traj2['phi'])
        y2 = traj2['r'] * np.sin(traj2['phi'])
        ax.plot(x2, y2, color='#d62728', linewidth=2.2, linestyle='--',
                label=f'Fragment 2  ($E_2$={opt["E2"]:.2f})', zorder=3)

        x3 = traj3['r'] * np.cos(traj3['phi'])
        y3 = traj3['r'] * np.sin(traj3['phi'])
        ax.plot(x3, y3, color='#2ca02c', linewidth=2.2, linestyle='-.',
                label=f'Fragment 3  ($E_3$={opt["E3"]:.2f})', zorder=3)

        # Split point (deepest penetration)
        x_s = traj['r'][deepest_idx] * np.cos(traj['phi'][deepest_idx])
        y_s = traj['r'][deepest_idx] * np.sin(traj['phi'][deepest_idx])
        ax.scatter([x_s], [y_s], color='black', s=150, zorder=5,
                   marker='X', edgecolors='white', linewidths=2.5,
                   label='Split point')

        ax.scatter([R0 * np.cos(PHI0)], [R0 * np.sin(PHI0)],
                   color='#1f77b4', s=100, zorder=5, marker='o',
                   edgecolors='white', linewidths=2.0)

        # Adaptive limits
        all_x = np.concatenate([x_in, x2, x3])
        all_y = np.concatenate([y_in, y2, y3])
        margin = 1.0
        max_range = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) + margin
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)

        escape_str = "ESCAPED ✓" if res['verified'] else "DID NOT ESCAPE ✗"
        
        if label == 'Static':
            ann = (
                f'{label} Kerr\n'
                f'$a/M = {A:.3f}$\n'
                f'$m = {res["m_split"]:.3f}M$\n'
                f'$r_s = {res["r_s"]:.3f}M$\n'
                f'$r_+ = {res["r_plus"]:.3f}M$\n'
                f'$p_{{r,3}} = {opt["p_r3"]:.2f}$\n'
                f'$\\eta = {opt["eta"]*100:.1f}\\%$\n'
                f'{escape_str}'
            )
        else:
            ann = (
                f'{label} Kerr-Vaidya\n'
                f'$a/M = {A:.3f}$\n'
                f'$m = {res["m_split"]:.3f}M$\n'
                f'$r_s = {res["r_s"]:.3f}M$\n'
                f'$r_+ = {res["r_plus"]:.3f}M$\n'
                f'$\\dot{{m}} = -{ALPHA:.4f}$\n'
                f'$p_{{r,3}} = {opt["p_r3"]:.2f}$\n'
                f'$\\eta = {opt["eta"]*100:.1f}\\%$\n'
                f'{escape_str}'
            )

        ax.text(0.02, 0.98, ann,
                transform=ax.transAxes, fontsize=10, va='top', ha='left',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='gray', alpha=0.95))

        ax.set_aspect('equal')
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

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for label in ['Static', 'Dynamic']:
        res = results[label]
        opt = res['opt']
        print(f"\n{label}:")
        print(f"  E1={res['E1']:.4f}, L1={res['L1']:.4f}")
        print(f"  E2={opt['E2']:.4f}, L2={opt['L2']:.4f}")
        print(f"  E3={opt['E3']:.4f}, L3={opt['L3']:.4f}")
        print(f"  p_r3={opt['p_r3']:.4f}")
        print(f"  Efficiency: eta={opt['eta']*100:.2f}%")
        print(f"  Fragment 3 final r: {res['traj3']['r'][-1]:.2f}M")
        print(f"  VERIFIED ESCAPE: {'YES' if res['verified'] else 'NO'}")

    eta_s = results['Static']['opt']['eta']
    eta_d = results['Dynamic']['opt']['eta']
    delta = (eta_d - eta_s) / eta_s * 100

    print(f"\n{'=' * 70}")
    print(f"KEY RESULT:")
    print(f"  Static efficiency:  {eta_s*100:.2f}%")
    print(f"  Dynamic efficiency: {eta_d*100:.2f}%")
    print(f"  Relative improvement: {delta:+.1f}%")
    print(f"{'=' * 70}")
    print("\nStep 04 complete.")
    print("=" * 70)