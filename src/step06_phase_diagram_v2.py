#!/usr/bin/env python3
"""
================================================================================
Step 06: Critical Mass-Loss Rate & Phase Diagram — FIXED v2
================================================================================

Fixes applied:
  1. Adaptive HORIZON_MIN_DIST: lower for low-spin BHs (thinner ergosphere).
  2. Higher ALPHA_MAX for high spins (a=0.9 was still viable at 0.01).
  3. Spins a < 0.8 use relaxed safety margin or are skipped with note.
  4. Better error handling for bracket validation failures.

Run:
    python src/step06_phase_diagram_v2.py
================================================================================
"""

import sys
import os
import json
import warnings

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

from kerr_vaidya_metric import (
    KerrVaidyaParams,
    integrate_geodesic,
    effective_potential,
    penrose_efficiency,
    max_penrose_efficiency_static_kerr,
)

# ==============================================================================
# Config
# ==============================================================================

M0 = 1.0
M_FLOOR = 0.95
R0 = 6.0
PHI0 = 0.0
R_ESCAPE = 5.0

DATA_DIR = os.path.join(project_root, 'data')
OUT_DIR = os.path.join(project_root, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, 'step06_phase_diagram_v2.png')
OUT_PDF = os.path.join(OUT_DIR, 'step06_phase_diagram_v2.pdf')
DPI = 300

RES_PRESETS = {
    'fast':   ((0.70, 0.95, 10), (1.5, 2.8, 10), (-0.7, -0.1, 8)),
    'medium': ((0.65, 0.95, 14), (1.4, 3.0, 14), (-0.8, -0.05, 12)),
}
RES_MODE = 'fast'

# Binary search parameters
ALPHA_MIN = 1e-5
ALPHA_MAX_DEFAULT = 0.02      # Was 0.01 — a=0.9 was still viable at 0.01
ALPHA_TOL = 2e-4
MAX_ITER = 10                 # Increased from 8

# ==============================================================================
# Metric factories
# ==============================================================================

def make_m_funcs(alpha, m0=M0, m_floor=M_FLOOR):
    def m_func(t):
        t = np.asarray(t)
        return np.maximum(m_floor, m0 - alpha * t)
    def m_dot_func(t):
        t = np.asarray(t)
        m = m0 - alpha * t
        return np.where(m > m_floor, -alpha, 0.0)
    return m_func, m_dot_func

def make_static_m_funcs(m0=M0):
    def m_func(t):
        t = np.asarray(t)
        return np.full_like(t, m0, dtype=float)
    def m_dot_func(t):
        t = np.asarray(t)
        return np.zeros_like(t, dtype=float)
    return m_func, m_dot_func

# ==============================================================================
# Core optimization
# ==============================================================================

def find_skimming_orbit(params, e_range, l_range, pr_range, label="", horizon_min_dist=0.30):
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
                if min_horizon_dist < horizon_min_dist:
                    continue
                depth = np.max(r_s_vals) - r_min
                score = depth * inside
                if score > best_score:
                    best_score = score
                    best = {'E': E, 'L': L, 'p_r0': p_r0, 'traj': traj,
                            'r_min': r_min, 'inside_fraction': inside / len(r),
                            'min_horizon_dist': min_horizon_dist}
    return best


def find_deepest_safe_point(traj, m_func, a, horizon_min_dist=0.30):
    r_s_vals = 2.0 * m_func(traj['t'])
    r_plus_vals = m_func(traj['t']) + np.sqrt(np.maximum(m_func(traj['t'])**2 - a**2, 0.0))
    inside_mask = traj['r'] <= r_s_vals
    if not np.any(inside_mask):
        return None
    inside_indices = np.where(inside_mask)[0]
    safe_mask = traj['r'][inside_indices] >= r_plus_vals[inside_indices] + horizon_min_dist
    if np.any(safe_mask):
        safe_indices = inside_indices[safe_mask]
        return safe_indices[np.argmin(traj['r'][safe_indices])]
    safe_all = traj['r'] >= r_plus_vals + horizon_min_dist
    safe_inside = inside_mask & safe_all
    if np.any(safe_inside):
        return np.where(safe_inside)[0][np.argmin(traj['r'][safe_inside])]
    return None


def brute_force_split(E1, L1, r_split, phi_split, t_split, params, label, a):
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


def test_alpha_viable(a, alpha, res_preset, label, horizon_min_dist=0.30):
    print(f"\n  [{label}] Testing a={a:.2f}, alpha={alpha:.5f}...")
    m_f, m_df = make_m_funcs(alpha, M0, M_FLOOR)
    params = KerrVaidyaParams(a=a, m_func=m_f, m_dot_func=m_df)
    candidate = find_skimming_orbit(params, *res_preset, label=label, horizon_min_dist=horizon_min_dist)
    if candidate is None:
        print(f"  [{label}] -> NO SKIMMING ORBIT (not viable)")
        return False, None
    E1, L1 = candidate['E'], candidate['L']
    traj_in = integrate_geodesic(
        r0=R0, phi0=PHI0, E=E1, L=L1, p_r0=candidate['p_r0'],
        tau_max=300.0, params=params, n_points=1500
    )
    deepest_idx = find_deepest_safe_point(traj_in, params.m_func, a, horizon_min_dist=horizon_min_dist)
    if deepest_idx is None:
        print(f"  [{label}] -> NO SAFE SPLIT POINT (not viable)")
        return False, None
    t_s = traj_in['t'][deepest_idx]
    r_s = traj_in['r'][deepest_idx]
    phi_s = traj_in['phi'][deepest_idx]
    opt = brute_force_split(E1, L1, r_s, phi_s, t_s, params, label, a)
    if opt is None:
        print(f"  [{label}] -> NO ESCAPE-VERIFIED SPLIT (not viable)")
        return False, None
    traj3 = integrate_geodesic(
        r0=r_s, phi0=phi_s, E=opt['E3'], L=opt['L3'], p_r0=opt['p_r3'],
        tau_max=600.0, params=params, n_points=2500
    )
    verified = bool(traj3['r'][-1] > R_ESCAPE)
    if verified:
        print(f"  [{label}] -> VIABLE, eta={opt['eta']*100:.2f}%")
        return True, float(opt['eta'])
    else:
        print(f"  [{label}] -> VERIFICATION FAILED (not viable)")
        return False, None


# ==============================================================================
# Adaptive horizon distance based on spin
# ==============================================================================

def get_horizon_min_dist(a):
    """
    Adaptive safety margin: thinner ergospheres need smaller margins.
    For a/M < 0.75, the ergosphere is very thin — use relaxed margin.
    """
    if a >= 0.85:
        return 0.30
    elif a >= 0.75:
        return 0.20
    elif a >= 0.65:
        return 0.15
    else:
        return 0.10


def get_alpha_max(a):
    """Higher spins can sustain higher alpha before shutting down."""
    if a >= 0.95:
        return 0.02
    elif a >= 0.90:
        return 0.015
    elif a >= 0.80:
        return 0.012
    else:
        return 0.008


# ==============================================================================
# Binary search for alpha_crit
# ==============================================================================

def find_alpha_crit(a, alpha_min, alpha_max, tol, max_iter, res_preset, horizon_min_dist):
    print(f"\n{'='*60}")
    print(f"Finding alpha_crit for a/M = {a:.2f}")
    print(f"  Horizon min dist: {horizon_min_dist:.2f}M")
    print(f"  Initial bracket: [{alpha_min:.5f}, {alpha_max:.5f}]")
    print(f"  Tolerance: {tol:.5f}, Max iterations: {max_iter}")
    print(f"{'='*60}")

    history = []
    lo, hi = alpha_min, alpha_max

    viable_lo, eta_lo = test_alpha_viable(a, lo, res_preset, f"a={a:.2f}_lo", horizon_min_dist)
    viable_hi, eta_hi = test_alpha_viable(a, hi, res_preset, f"a={a:.2f}_hi", horizon_min_dist)

    if not viable_lo:
        print(f"  WARNING: alpha_min={alpha_min} is already non-viable!")
        # Try with even smaller alpha
        test_alpha = alpha_min / 10.0
        viable_test, eta_test = test_alpha_viable(a, test_alpha, res_preset, f"a={a:.2f}_test", horizon_min_dist)
        if viable_test:
            print(f"  Found viable at alpha={test_alpha:.6f}, using as new lo")
            lo = test_alpha
            viable_lo = True
            eta_lo = eta_test
        else:
            return {'a': float(a), 'alpha_crit': None, 'status': 'alpha_min_nonviable',
                    'history': history, 'lo': float(lo), 'hi': float(hi)}

    if viable_hi:
        print(f"  WARNING: alpha_max={alpha_max} is still viable! Increasing...")
        # Try higher alpha
        for multiplier in [1.5, 2.0, 3.0, 5.0]:
            test_alpha = alpha_max * multiplier
            viable_test, eta_test = test_alpha_viable(a, test_alpha, res_preset, f"a={a:.2f}_hi{multiplier}", horizon_min_dist)
            if not viable_test:
                hi = test_alpha
                viable_hi = False
                eta_hi = None
                print(f"  Found non-viable at alpha={test_alpha:.5f}")
                break
        if viable_hi:
            return {'a': float(a), 'alpha_crit': None, 'status': 'alpha_max_still_viable',
                    'history': history, 'lo': float(lo), 'hi': float(hi)}

    history.append({'iter': 0, 'alpha': float(lo), 'viable': bool(viable_lo), 'eta': eta_lo, 'lo': float(lo), 'hi': float(hi)})
    history.append({'iter': 0, 'alpha': float(hi), 'viable': bool(viable_hi), 'eta': eta_hi, 'lo': float(lo), 'hi': float(hi)})

    for iteration in range(1, max_iter + 1):
        mid = (lo + hi) / 2.0
        viable_mid, eta_mid = test_alpha_viable(a, mid, res_preset, f"a={a:.2f}_i{iteration}", horizon_min_dist)
        history.append({'iter': iteration, 'alpha': float(mid), 'viable': bool(viable_mid),
                        'eta': eta_mid, 'lo': float(lo), 'hi': float(hi)})

        if viable_mid:
            lo = mid
        else:
            hi = mid

        width = hi - lo
        print(f"  Iter {iteration}: bracket = [{lo:.5f}, {hi:.5f}], width = {width:.5f}")

        if width < tol:
            print(f"  CONVERGED: alpha_crit ≈ {lo:.5f} (width < {tol})")
            break

    alpha_crit = lo
    print(f"\n  RESULT: alpha_crit(a={a:.2f}) = {alpha_crit:.5f}")
    return {
        'a': float(a),
        'alpha_crit': float(alpha_crit),
        'status': 'converged',
        'history': history,
        'final_lo': float(lo),
        'final_hi': float(hi),
        'n_iterations': iteration,
        'horizon_min_dist': float(horizon_min_dist),
    }


# ==============================================================================
# Main phase diagram computation
# ==============================================================================

def compute_phase_boundary():
    print("=" * 70)
    print("Step 06: Critical Mass-Loss Rate & Phase Diagram — FIXED v2")
    print(f"Resolution mode: {RES_MODE}")
    print("=" * 70)

    spins = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    results = []

    for a in spins:
        hmd = get_horizon_min_dist(a)
        amax = get_alpha_max(a)
        res = find_alpha_crit(a, ALPHA_MIN, amax, ALPHA_TOL, MAX_ITER, RES_PRESETS[RES_MODE], hmd)
        results.append(res)
        with open(os.path.join(DATA_DIR, 'section_phase_boundary_v2.json'), 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else (bool(x) if isinstance(x, np.bool_) else str(x)))

    return results


# ==============================================================================
# Plot phase diagram
# ==============================================================================

def plot_phase_diagram(boundary_results):
    print("\n" + "=" * 70)
    print("Generating phase diagram...")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    converged = [r for r in boundary_results if r.get('alpha_crit') is not None]
    a_vals = [r['a'] for r in converged]
    alpha_crits = [r['alpha_crit'] for r in converged]

    # --- Panel A: Phase diagram (a, alpha) ---
    ax_a = fig.add_subplot(gs[0, :])

    a_theory = np.linspace(0.45, 0.99, 200)
    thickness = M0 - np.sqrt(np.maximum(M0**2 - a_theory**2, 0.0))
    if alpha_crits:
        norm = alpha_crits[-1] / (thickness[-1] ** 1.5 + 1e-10)
        alpha_theory_est = norm * thickness ** 1.5
    else:
        alpha_theory_est = thickness * 0.001

    ax_a.fill_between(a_theory, 0, alpha_theory_est, alpha=0.15, color='green',
                      label='Viable extraction region (estimate)')
    ax_a.fill_between(a_theory, alpha_theory_est, 0.025, alpha=0.15, color='red',
                      label='Non-viable region (estimate)')

    if converged:
        ax_a.plot(a_vals, alpha_crits, 'ko-', linewidth=2.5, markersize=10,
                  markerfacecolor='#ff7f0e', markeredgewidth=2, zorder=5,
                  label='Numerical boundary (binary search)')
        for a, ac in zip(a_vals, alpha_crits):
            ax_a.annotate(f'$\\alpha_{{\\rm crit}} = {ac:.4f}$',
                          xy=(a, ac), xytext=(a + 0.02, ac + 0.0005),
                          fontsize=10, fontweight='bold',
                          arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax_a.set_xlabel('Spin parameter $a/M$', fontsize=14)
    ax_a.set_ylabel('Critical mass-loss rate $\\alpha_{\\rm crit} = -\\dot{m}_{\\rm crit}$', fontsize=14)
    ax_a.set_title('(a) Phase Diagram: Viable vs. Non-Viable Penrose Extraction', fontsize=14, fontweight='bold')
    ax_a.set_xlim(0.45, 1.0)
    ax_a.set_ylim(0, 0.025)
    ax_a.legend(fontsize=11, loc='upper left')
    ax_a.grid(True, alpha=0.3)

    # --- Panel B: alpha_crit vs. spin ---
    ax_b = fig.add_subplot(gs[1, 0])
    if converged:
        ax_b.plot(a_vals, alpha_crits, 's-', color='#1f77b4', linewidth=2.5, markersize=10,
                  markerfacecolor='white', markeredgewidth=2)
        ax_b.set_xlabel('$a/M$', fontsize=13)
        ax_b.set_yscale('log')
        ax_b.set_ylabel('$\\alpha_{\\rm crit}$ (log scale)', fontsize=13)
        ax_b.set_title('(b) Critical Rate vs. Spin (log)', fontsize=13, fontweight='bold')
        ax_b.grid(True, alpha=0.3, which='both')

    # --- Panel C: Efficiency at alpha_crit ---
    ax_c = fig.add_subplot(gs[1, 1])
    alpha_sweep_path = os.path.join(DATA_DIR, 'section_alpha.json')
    if os.path.exists(alpha_sweep_path):
        with open(alpha_sweep_path, 'r') as f:
            alpha_data = json.load(f)
        alphas = [d['alpha'] for d in alpha_data]
        etas = [d['eta'] * 100 for d in alpha_data]
        ax_c.plot(alphas, etas, 'o-', color='#2ca02c', linewidth=2, markersize=8,
                  markerfacecolor='white', markeredgewidth=2, label='$a/M = 0.9$')
        if converged and a_vals:
            a_target = 0.9
            closest_idx = np.argmin(np.abs(np.array(a_vals) - a_target))
            ac_target = alpha_crits[closest_idx]
            ax_c.axvline(x=ac_target, color='red', linestyle='--', linewidth=2,
                        label=f'$\\alpha_{{\\rm crit}} = {ac_target:.4f}$')
        ax_c.set_xlabel('$\\alpha$', fontsize=13)
        ax_c.set_ylabel('$\\eta$ [%]', fontsize=13)
        ax_c.set_title('(c) Efficiency Vanishing at $\\alpha_{\\rm crit}$', fontsize=13, fontweight='bold')
        ax_c.legend(fontsize=10)
        ax_c.grid(True, alpha=0.3)
    else:
        ax_c.text(0.5, 0.5, 'Run Step 05 first\nfor alpha sweep data',
                  ha='center', va='center', transform=ax_c.transAxes, fontsize=12)

    fig.suptitle(
        'Phase Boundary for Penrose Energy Extraction from Radiating Kerr-Vaidya Black Holes',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUT_PDF, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\nSaved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    boundary_results = compute_phase_boundary()
    plot_phase_diagram(boundary_results)
    print("\n" + "=" * 70)
    print("Step 06 complete.")
    print("=" * 70)