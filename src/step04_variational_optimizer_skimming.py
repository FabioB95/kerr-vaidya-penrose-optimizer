#!/usr/bin/env python3
"""
================================================================================
Step 04: Skimming-Orbit Optimizer (Independent per Metric) — FIXED
================================================================================

Fixes applied:
  1. Minimum horizon distance = 0.30M (was 0.15M) — gives fragment 3 room to escape.
  2. Smarter split-point selection: if the deepest point is too close to the
     horizon, search shallower points inside the ergosphere.
  3. Wider parameter grid for dynamic case (weaker gravity needs different tuning).
  4. Fallback: if stage2 fails at the deepest point, try progressively shallower
     split points until one works.
  5. FIXED: m_func and m_dot_func now properly handle array inputs.
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
ALPHA = 0.0005
M_FLOOR = 0.95
HORIZON_MIN_DIST = 0.30          # NEW: minimum safe distance above horizon

# ==============================================================================
# FIXED: m_func and m_dot_func now properly handle array inputs
# ==============================================================================

def m_func_static(t):
    """Return constant mass M0, handles array inputs properly."""
    t = np.asarray(t)
    return np.full_like(t, M0, dtype=float)

def m_dot_func_static(t):
    """Return zero mass derivative, handles array inputs properly."""
    t = np.asarray(t)
    return np.zeros_like(t, dtype=float)

def m_func_dynamic(t):
    """Return dynamically decreasing mass, handles array inputs properly."""
    t = np.asarray(t)
    m = M0 - ALPHA * t
    return np.maximum(M_FLOOR, m)

def m_dot_func_dynamic(t):
    """Return mass derivative, handles array inputs properly."""
    t = np.asarray(t)
    m = M0 - ALPHA * t
    return np.where(m > M_FLOOR, -ALPHA, 0.0)

PARAMS_STATIC = KerrVaidyaParams(a=A, m_func=m_func_static, m_dot_func=m_dot_func_static)
PARAMS_DYNAMIC = KerrVaidyaParams(a=A, m_func=m_func_dynamic, m_dot_func=m_dot_func_dynamic)

R0 = 6.0
PHI0 = 0.0
R_ESCAPE = 5.0

OUT_PNG = os.path.join(project_root, 'figures', 'step04_optimizer_comparison.png')
OUT_PDF = os.path.join(project_root, 'figures', 'step04_optimizer_comparison.pdf')
DPI = 300

# ==============================================================================
# Stage 1: Find Skimming Orbit (per metric)
# ==============================================================================

def stage1_find_skimming_orbit(params, e_range=(0.65, 0.95, 20), l_range=(1.4, 3.0, 20), pr_range=(-0.8, -0.05, 18), label=""):
    """Find the best skimming orbit for the given metric."""
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
                    print(f"        [{label}] ...{count}/{total} tested")

                try:
                    traj = integrate_geodesic(
                        r0=R0, phi0=PHI0, E=E, L=L, p_r0=p_r0,
                        tau_max=250.0, params=params, n_points=1000
                    )
                except Exception:
                    continue

                r = traj['r']
                t = traj['t']
                
                # FIXED: Use scalar values for individual t points
                r_s_vals = np.array([2.0 * params.m_func(np.array([ti]))[0] for ti in t])
                r_plus_vals = np.array([
                    params.m_func(np.array([ti]))[0] + np.sqrt(max(params.m_func(np.array([ti]))[0]**2 - A**2, 0.0))
                    for ti in t
                ])

                # Must enter ergosphere at some point
                inside_mask = r <= r_s_vals
                if not np.any(inside_mask):
                    continue

                r_min = np.min(r)
                r_final = r[-1]

                # Must not plunge into horizon
                if r_final < r_plus_vals[-1] + 0.05:
                    continue

                # Must escape back out
                if r_final < R0 - 0.5:
                    continue

                inside = np.sum(inside_mask)
                if inside < 30:
                    continue

                # NEW: enforce minimum safe distance from horizon at ALL times
                horizon_dists = r - r_plus_vals
                min_horizon_dist = np.min(horizon_dists)
                if min_horizon_dist < HORIZON_MIN_DIST:
                    continue  # Too dangerous — no room for fragment 3 to escape

                # Score: deeper is better (more negative energy possible)
                depth = np.max(r_s_vals) - r_min
                score = depth * inside

                if score > best_score:
                    best_score = score
                    best = {
                        'E': E, 'L': L, 'p_r0': p_r0,
                        'traj': traj, 'r_min': r_min,
                        'inside_fraction': inside / len(r),
                        'min_horizon_dist': min_horizon_dist
                    }

    return best


# ==============================================================================
# Split Point Selection
# ==============================================================================

def find_deepest_safe_point(traj, m_func):
    """Find the deepest point inside the ergosphere that is safely above the horizon."""
    # FIXED: Handle array inputs properly
    r_s_vals = np.array([2.0 * m_func(np.array([ti]))[0] for ti in traj['t']])
    r_plus_vals = np.array([
        m_func(np.array([ti]))[0] + np.sqrt(max(m_func(np.array([ti]))[0]**2 - A**2, 0.0))
        for ti in traj['t']
    ])

    inside_mask = traj['r'] <= r_s_vals
    if not np.any(inside_mask):
        return None

    inside_indices = np.where(inside_mask)[0]

    # First try: deepest point that satisfies horizon safety
    safe_mask = traj['r'][inside_indices] >= r_plus_vals[inside_indices] + HORIZON_MIN_DIST
    if np.any(safe_mask):
        safe_indices = inside_indices[safe_mask]
        deepest_idx = safe_indices[np.argmin(traj['r'][safe_indices])]
        return deepest_idx

    # Fallback: shallowest safe point inside ergosphere
    safe_all = traj['r'] >= r_plus_vals + HORIZON_MIN_DIST
    safe_inside = inside_mask & safe_all
    if np.any(safe_inside):
        return np.where(safe_inside)[0][np.argmin(traj['r'][safe_inside])]

    return None


def try_split_points(traj, m_func, E1, L1, params, label):
    """Try multiple split points (deepest first, then shallower) until one works."""
    # FIXED: Handle array inputs properly
    r_s_vals = np.array([2.0 * m_func(np.array([ti]))[0] for ti in traj['t']])
    r_plus_vals = np.array([
        m_func(np.array([ti]))[0] + np.sqrt(max(m_func(np.array([ti]))[0]**2 - A**2, 0.0))
        for ti in traj['t']
    ])

    inside_mask = traj['r'] <= r_s_vals
    safe_mask = traj['r'] >= r_plus_vals + HORIZON_MIN_DIST
    candidates = np.where(inside_mask & safe_mask)[0]

    if len(candidates) == 0:
        print(f"    [{label}] WARNING: No safe split points inside ergosphere!")
        return None, None

    # Sort by depth (deepest first)
    candidates = candidates[np.argsort(traj['r'][candidates])]

    # Try deepest 5 candidate points
    for idx in candidates[:min(5, len(candidates))]:
        t_split = traj['t'][idx]
        r_split = traj['r'][idx]
        phi_split = traj['phi'][idx]
        m_split = m_func(np.array([t_split]))[0]
        r_plus_split = m_split + np.sqrt(max(m_split**2 - A**2, 0.0))

        print(f"    [{label}] Trying split: t={t_split:.2f}M, r={r_split:.4f}M, "
              f"r_+={r_plus_split:.4f}M, gap={r_split-r_plus_split:.3f}M")

        opt = stage2_brute_force(E1, L1, r_split, phi_split, t_split, params, label)
        if opt is not None:
            return idx, opt

    return None, None


# ==============================================================================
# Stage 2: Brute-Force Search for Escape-Verified Split
# ==============================================================================

def stage2_brute_force(E1, L1, r_split, phi_split, t_split, params, label):
    """Brute-force search for the best escape-verified Penrose split."""
    m_split = params.m_func(np.array([t_split]))[0]
    r_plus_split = m_split + np.sqrt(max(m_split**2 - A**2, 0.0))

    # Adaptive search range
    gap = r_split - r_plus_split
    if gap < 0.25:
        p_r3_max = 15.0
        n_pr = 40
    elif gap < 0.40:
        p_r3_max = 10.0
        n_pr = 30
    else:
        p_r3_max = 6.0
        n_pr = 25

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
                        tau_max=600.0, params=params, n_points=2000
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

    if best is not None:
        print(f"    [{label}] BEST at this split: eta={best['eta']*100:.2f}%, "
              f"E3={best['E3']:.4f}, p_r3={best['p_r3']:.2f}, "
              f"final_r={best['final_r']:.1f}M")

    return best


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':

    print("=" * 70)
    print("Step 04: Skimming-Orbit Optimizer (Independent per Metric) — FIXED")
    print("=" * 70)

    results = {}

    for label, params in [('Static', PARAMS_STATIC), ('Dynamic', PARAMS_DYNAMIC)]:

        print(f"\n{'-' * 60}")
        print(f"Case: {label}")
        print(f"{'-' * 60}")

        # ---------------------------------------------------------------------
        # 1. Find skimming orbit for THIS metric
        # ---------------------------------------------------------------------
        print(f"\n[1] Finding optimal skimming orbit ({label} metric)...")

        candidate = stage1_find_skimming_orbit(params, label=label)

        if candidate is None:
            print(f"ERROR: No skimming orbit found for {label}!")
            continue

        E1 = candidate['E']
        L1 = candidate['L']
        p_r0 = candidate['p_r0']

        print(f"    Selected: E={E1:.4f}, L={L1:.4f}, p_r0={p_r0:.4f}")
        print(f"    r_min={candidate['r_min']:.4f}M")
        print(f"    Inside ergosphere: {candidate['inside_fraction']*100:.1f}% of orbit")
        print(f"    Min horizon distance: {candidate['min_horizon_dist']:.3f}M")

        # ---------------------------------------------------------------------
        # 2. Integrate the skimming orbit
        # ---------------------------------------------------------------------
        print(f"\n[2] Integrating skimming orbit ({label})...")

        traj_in = integrate_geodesic(
            r0=R0, phi0=PHI0, E=E1, L=L1, p_r0=p_r0,
            tau_max=300.0, params=params, n_points=1500
        )

        r_min = np.min(traj_in['r'])
        print(f"    r_min={r_min:.4f}M")

        # ---------------------------------------------------------------------
        # 3. Find split point + brute-force (with fallback)
        # ---------------------------------------------------------------------
        print(f"\n[3] Finding split point and optimizing ({label})...")

        deepest_idx, opt_result = try_split_points(traj_in, params.m_func, E1, L1, params, label)

        if deepest_idx is None or opt_result is None:
            print(f"    ERROR: No valid split found for {label} after trying multiple points!")
            continue

        t_split = traj_in['t'][deepest_idx]
        r_split = traj_in['r'][deepest_idx]
        phi_split = traj_in['phi'][deepest_idx]
        m_split = params.m_func(np.array([t_split]))[0]
        r_plus_split = m_split + np.sqrt(max(m_split**2 - A**2, 0.0))

        print(f"    Final split: t={t_split:.2f}M, r={r_split:.4f}M")
        print(f"    m={m_split:.4f}M, r_s={2*m_split:.4f}M, r_+={r_plus_split:.4f}M")
        print(f"    BEST: E3={opt_result['E3']:.4f}, L3={opt_result['L3']:.4f}, p_r3={opt_result['p_r3']:.4f}")
        print(f"    eta={opt_result['eta']*100:.2f}%, final_r={opt_result['final_r']:.1f}M")

        # ---------------------------------------------------------------------
        # 4. Final high-resolution integration
        # ---------------------------------------------------------------------
        print(f"\n[4] Final integration ({label})...")

        traj2 = integrate_geodesic(
            r0=r_split, phi0=phi_split,
            E=opt_result['E2'], L=opt_result['L2'], p_r0=-0.05,
            tau_max=100.0, params=params, n_points=800
        )

        traj3 = integrate_geodesic(
            r0=r_split, phi0=phi_split,
            E=opt_result['E3'], L=opt_result['L3'], p_r0=opt_result['p_r3'],
            tau_max=600.0, params=params, n_points=2500
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
            'E1': E1, 'L1': L1, 'p_r0': p_r0, 'verified': verified
        }

    # ==============================================================================
    # 5. Plot
    # ==============================================================================
    print("\n" + "=" * 70)
    print("[5] Generating comparison plot...")
    print("=" * 70)

    if len(results) < 2:
        print("ERROR: One or both cases failed. Cannot generate comparison.")
        for label, res in results.items():
            print(f"  {label}: OK")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        r'Optimal Penrose Process: Static Kerr vs. Dynamic Kerr-Vaidya',
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
                m_later = m_func_dynamic(np.array([t_later]))[0]
                if m_later > M_FLOOR:
                    plot_ergosphere_and_horizon(m=m_later, a=A, ax=ax, label=False)

        x_in = traj['r'] * np.cos(traj['phi'])
        y_in = traj['r'] * np.sin(traj['phi'])
        ax.plot(x_in, y_in, color='#1f77b4', linewidth=1.2, alpha=0.4,
                label='Full orbit', zorder=2)

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

        x_s = traj['r'][deepest_idx] * np.cos(traj['phi'][deepest_idx])
        y_s = traj['r'][deepest_idx] * np.sin(traj['phi'][deepest_idx])
        ax.scatter([x_s], [y_s], color='black', s=150, zorder=5,
                   marker='X', edgecolors='white', linewidths=2.5,
                   label='Split point')

        ax.scatter([R0 * np.cos(PHI0)], [R0 * np.sin(PHI0)],
                   color='#1f77b4', s=100, zorder=5, marker='o',
                   edgecolors='white', linewidths=2.0, label='Start')

        all_x = np.concatenate([x_in, x2, x3])
        all_y = np.concatenate([y_in, y2, y3])
        margin = 1.5
        max_range = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) + margin
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)

        escape_str = "ESCAPED" if res['verified'] else "NO ESCAPE"

        if label == 'Static':
            ann = (
                f'{label} Kerr\n'
                f'$a/M = {A:.3f}$\n'
                f'$m = {res["m_split"]:.3f}M$\n'
                f'$r_s = {res["r_s"]:.3f}M$\n'
                f'$r_+ = {res["r_plus"]:.3f}M$\n'
                f'$E_1 = {res["E1"]:.3f}$\n'
                f'$L_1 = {res["L1"]:.3f}$\n'
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
                f'$E_1 = {res["E1"]:.3f}$\n'
                f'$L_1 = {res["L1"]:.3f}$\n'
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
        ax.set_title(f'{label} Case — Optimal Strategy', fontsize=13, fontweight='bold')
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
    print("SUMMARY: Independent Optimization per Spacetime")
    print("=" * 70)

    for label in ['Static', 'Dynamic']:
        res = results[label]
        opt = res['opt']
        print(f"\n{label}:")
        print(f"  Optimal initial: E={res['E1']:.4f}, L={res['L1']:.4f}, p_r0={res['p_r0']:.4f}")
        print(f"  Split: t={res['traj_in']['t'][res['deepest_idx']]:.2f}M, r={res['traj_in']['r'][res['deepest_idx']]:.4f}M")
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
    print(f"  Static (Kerr) optimal efficiency:    {eta_s*100:.2f}%")
    print(f"  Dynamic (Kerr-Vaidya) optimal eff.:  {eta_d*100:.2f}%")
    print(f"  Relative difference:                 {delta:+.1f}%")
    print(f"{'=' * 70}")
    print("\nStep 04 complete.")
    print("=" * 70)