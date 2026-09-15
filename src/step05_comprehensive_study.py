#!/usr/bin/env python3
"""
================================================================================
Step 05: Comprehensive Parametric Study & Publication Figure
================================================================================

A. Mass-loss rate sweep (alpha = 1e-4 ... 5e-3)
B. Spin parameter sweep (a/M = 0.5 ... 0.99)
C. Time-resolved efficiency tracking
D. Convergence & validation study
E. Theoretical bound comparison
F. Publication-quality 3x2 multi-panel figure

Run:
    python src/step05_comprehensive_study.py
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
    event_horizon_radius,
    effective_potential,
    penrose_efficiency,
    plot_ergosphere_and_horizon,
    max_penrose_efficiency_static_kerr,
)

# ==============================================================================
# JSON helper: convert numpy types to native Python
# ==============================================================================

def _clean(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _clean(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v

# ==============================================================================
# Configuration
# ==============================================================================

M0 = 1.0
A_DEFAULT = 0.9
ALPHA_DEFAULT = 0.0005
M_FLOOR = 0.95
HORIZON_MIN_DIST = 0.30
R0 = 6.0
PHI0 = 0.0
R_ESCAPE = 5.0

OUT_DIR = os.path.join(project_root, 'figures')
DATA_DIR = os.path.join(project_root, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, 'step05_comprehensive_study.png')
OUT_PDF = os.path.join(OUT_DIR, 'step05_comprehensive_study.pdf')
DPI = 300

RES_PRESETS = {
    'fast':   ((0.70, 0.95, 10), (1.5, 2.8, 10), (-0.7, -0.1, 8)),
    'medium': ((0.65, 0.95, 14), (1.4, 3.0, 14), (-0.8, -0.05, 12)),
    'fine':   ((0.65, 0.95, 20), (1.4, 3.0, 20), (-0.8, -0.05, 18)),
}
RES_MODE = 'medium'

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

def find_skimming_orbit(params, e_range, l_range, pr_range, label=""):
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


def optimize_single_case(params, label, a, res_preset):
    print(f"\n  [{label}] Stage 1: skimming search...")
    candidate = find_skimming_orbit(params, *res_preset, label=label)
    if candidate is None:
        print(f"  [{label}] NO SKIMMING ORBIT FOUND")
        return None
    E1, L1, p_r0 = candidate['E'], candidate['L'], candidate['p_r0']
    print(f"  [{label}] E1={E1:.4f} L1={L1:.4f} p_r0={p_r0:.4f} r_min={candidate['r_min']:.4f}")
    traj_in = integrate_geodesic(
        r0=R0, phi0=PHI0, E=E1, L=L1, p_r0=p_r0,
        tau_max=300.0, params=params, n_points=1500
    )
    deepest_idx = find_deepest_safe_point(traj_in, params.m_func, a)
    if deepest_idx is None:
        print(f"  [{label}] NO SAFE SPLIT POINT")
        return None
    t_s = traj_in['t'][deepest_idx]
    r_s = traj_in['r'][deepest_idx]
    phi_s = traj_in['phi'][deepest_idx]
    print(f"  [{label}] Split at t={t_s:.2f} r={r_s:.4f}")
    opt = brute_force_split(E1, L1, r_s, phi_s, t_s, params, label, a)
    if opt is None:
        print(f"  [{label}] NO ESCAPE-VERIFIED SPLIT")
        return None
    traj3 = integrate_geodesic(
        r0=r_s, phi0=phi_s, E=opt['E3'], L=opt['L3'], p_r0=opt['p_r3'],
        tau_max=600.0, params=params, n_points=2500
    )
    verified = bool(traj3['r'][-1] > R_ESCAPE)
    print(f"  [{label}] eta={opt['eta']*100:.2f}% verified={verified}")
    return {
        'E1': float(E1), 'L1': float(L1), 'p_r0': float(p_r0),
        't_split': float(t_s), 'r_split': float(r_s), 'phi_split': float(phi_s),
        'E2': float(opt['E2']), 'L2': float(opt['L2']),
        'E3': float(opt['E3']), 'L3': float(opt['L3']), 'p_r3': float(opt['p_r3']),
        'eta': float(opt['eta']), 'verified': verified,
        'm_split': float(params.m_func(np.array([t_s]))[0]),
        'r_min': float(candidate['r_min']),
    }


# ==============================================================================
# A. Mass-loss rate sweep
# ==============================================================================

def section_alpha_sweep(a=A_DEFAULT):
    print("\n" + "=" * 70)
    print("SECTION A: Mass-Loss Rate Sweep")
    print("=" * 70)
    alphas = [0.0, 0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005]
    results = []
    for alpha in alphas:
        label = f"alpha={alpha:.4f}"
        print(f"\n--- alpha = {alpha:.4f} ---")
        if alpha == 0.0:
            m_f, m_df = make_static_m_funcs(M0)
        else:
            m_f, m_df = make_m_funcs(alpha, M0, M_FLOOR)
        params = KerrVaidyaParams(a=a, m_func=m_f, m_dot_func=m_df)
        res = optimize_single_case(params, label, a, RES_PRESETS[RES_MODE])
        if res is not None:
            res['alpha'] = float(alpha)
            results.append(res)
    with open(os.path.join(DATA_DIR, 'section_alpha.json'), 'w') as f:
        json.dump(_clean(results), f, indent=2)
    return results


# ==============================================================================
# B. Spin parameter sweep
# ==============================================================================

def section_spin_sweep(alpha=ALPHA_DEFAULT):
    print("\n" + "=" * 70)
    print("SECTION B: Spin Parameter Sweep")
    print("=" * 70)
    spins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    results_static = []
    results_dynamic = []
    for a in spins:
        label_s = f"static_a={a:.2f}"
        label_d = f"dynamic_a={a:.2f}"
        print(f"\n--- a/M = {a:.2f} ---")
        m_f, m_df = make_static_m_funcs(M0)
        params_s = KerrVaidyaParams(a=a, m_func=m_f, m_dot_func=m_df)
        res_s = optimize_single_case(params_s, label_s, a, RES_PRESETS[RES_MODE])
        if res_s is not None:
            res_s['a'] = float(a)
            results_static.append(res_s)
        m_f, m_df = make_m_funcs(alpha, M0, M_FLOOR)
        params_d = KerrVaidyaParams(a=a, m_func=m_f, m_dot_func=m_df)
        res_d = optimize_single_case(params_d, label_d, a, RES_PRESETS[RES_MODE])
        if res_d is not None:
            res_d['a'] = float(a)
            results_dynamic.append(res_d)
    with open(os.path.join(DATA_DIR, 'section_spin.json'), 'w') as f:
        json.dump(_clean({'static': results_static, 'dynamic': results_dynamic}), f, indent=2)
    return results_static, results_dynamic


# ==============================================================================
# C. Time-resolved efficiency
# ==============================================================================

def section_time_resolved(a=A_DEFAULT, alpha=ALPHA_DEFAULT):
    print("\n" + "=" * 70)
    print("SECTION C: Time-Resolved Efficiency")
    print("=" * 70)
    m_f, m_df = make_m_funcs(alpha, M0, M_FLOOR)
    params = KerrVaidyaParams(a=a, m_func=m_f, m_dot_func=m_df)
    candidate = find_skimming_orbit(params, *RES_PRESETS[RES_MODE], label="dynamic")
    if candidate is None:
        print("  No skimming orbit found!")
        return None
    E1, L1 = candidate['E'], candidate['L']
    traj = integrate_geodesic(
        r0=R0, phi0=PHI0, E=E1, L=L1, p_r0=candidate['p_r0'],
        tau_max=300.0, params=params, n_points=1500
    )
    r_s_vals = 2.0 * params.m_func(traj['t'])
    r_plus_vals = params.m_func(traj['t']) + np.sqrt(np.maximum(params.m_func(traj['t'])**2 - a**2, 0.0))
    inside_mask = (traj['r'] <= r_s_vals) & (traj['r'] >= r_plus_vals + HORIZON_MIN_DIST)
    indices = np.where(inside_mask)[0]
    indices = indices[::max(1, len(indices)//40)]
    time_data = []
    for idx in indices:
        t_s = traj['t'][idx]
        r_s = traj['r'][idx]
        phi_s = traj['phi'][idx]
        m_s = params.m_func(np.array([t_s]))[0]
        opt = brute_force_split(E1, L1, r_s, phi_s, t_s, params, f"t={t_s:.1f}", a)
        if opt is not None:
            time_data.append({
                't': float(t_s), 'r': float(r_s), 'm': float(m_s),
                'eta': float(opt['eta']), 'E3': float(opt['E3']), 'p_r3': float(opt['p_r3']),
                'gap': float(r_s - (m_s + np.sqrt(max(m_s**2 - a**2, 0.0))))
            })
    with open(os.path.join(DATA_DIR, 'section_time.json'), 'w') as f:
        json.dump(_clean(time_data), f, indent=2)
    return time_data


# ==============================================================================
# D. Convergence study
# ==============================================================================

def section_convergence(a=A_DEFAULT):
    print("\n" + "=" * 70)
    print("SECTION D: Convergence & Validation")
    print("=" * 70)
    m_f, m_df = make_static_m_funcs(M0)
    params = KerrVaidyaParams(a=a, m_func=m_f, m_dot_func=m_df)
    base = optimize_single_case(params, "base", a, RES_PRESETS[RES_MODE])
    if base is None:
        return None
    n_points_list = [500, 800, 1200, 1500, 2000, 3000]
    conv_n = []
    E1, L1, p_r0 = base['E1'], base['L1'], base['p_r0']
    for n_pts in n_points_list:
        traj = integrate_geodesic(r0=R0, phi0=PHI0, E=E1, L=L1, p_r0=p_r0,
                                  tau_max=300.0, params=params, n_points=n_pts)
        deepest_idx = find_deepest_safe_point(traj, params.m_func, a)
        if deepest_idx is None:
            continue
        t_s = traj['t'][deepest_idx]
        r_s = traj['r'][deepest_idx]
        phi_s = traj['phi'][deepest_idx]
        opt = brute_force_split(E1, L1, r_s, phi_s, t_s, params, f"n={n_pts}", a)
        if opt is not None:
            conv_n.append({'n_points': int(n_pts), 'eta': float(opt['eta']), 'E3': float(opt['E3'])})
            print(f"  n_points={n_pts:4d} -> eta={opt['eta']*100:.2f}%")
    grid_labels = ['fast', 'medium', 'fine']
    conv_grid = []
    for label in grid_labels:
        res = optimize_single_case(params, label, a, RES_PRESETS[label])
        if res is not None:
            conv_grid.append({'mode': label, 'eta': float(res['eta']), 'E1': float(res['E1'])})
            print(f"  grid={label:6s} -> eta={res['eta']*100:.2f}%")
    with open(os.path.join(DATA_DIR, 'section_conv.json'), 'w') as f:
        json.dump(_clean({'n_points_sweep': conv_n, 'grid_sweep': conv_grid}), f, indent=2)
    return conv_n, conv_grid


# ==============================================================================
# E. Theoretical bound comparison
# ==============================================================================

def section_theory(a=A_DEFAULT, alpha=ALPHA_DEFAULT):
    print("\n" + "=" * 70)
    print("SECTION E: Theoretical Bound Comparison")
    print("=" * 70)
    eta_theory = max_penrose_efficiency_static_kerr(M0, a)
    print(f"  Theoretical max (static Kerr, a={a}): eta = {eta_theory*100:.2f}%")
    m_f, m_df = make_static_m_funcs(M0)
    params_s = KerrVaidyaParams(a=a, m_func=m_f, m_dot_func=m_df)
    res_s = optimize_single_case(params_s, "static", a, RES_PRESETS[RES_MODE])
    m_f, m_df = make_m_funcs(alpha, M0, M_FLOOR)
    params_d = KerrVaidyaParams(a=a, m_func=m_f, m_dot_func=m_df)
    res_d = optimize_single_case(params_d, "dynamic", a, RES_PRESETS[RES_MODE])
    theory_data = {
        'a': float(a), 'alpha': float(alpha),
        'eta_theory_static': float(eta_theory),
        'eta_numerical_static': float(res_s['eta']) if res_s else None,
        'eta_numerical_dynamic': float(res_d['eta']) if res_d else None,
    }
    with open(os.path.join(DATA_DIR, 'section_theory.json'), 'w') as f:
        json.dump(_clean(theory_data), f, indent=2)
    return theory_data


# ==============================================================================
# F. Publication Figure
# ==============================================================================

def plot_publication_figure(alpha_data, spin_static, spin_dynamic, time_data, conv_n, conv_grid, theory_data):
    print("\n" + "=" * 70)
    print("SECTION F: Generating publication figure...")
    print("=" * 70)
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30)

    # Panel A: Efficiency vs. alpha
    ax_a = fig.add_subplot(gs[0, 0])
    if alpha_data:
        alphas = [d['alpha'] for d in alpha_data]
        etas = [d['eta'] * 100 for d in alpha_data]
        ax_a.plot(alphas, etas, 'o-', color='#1f77b4', linewidth=2, markersize=8,
                  markerfacecolor='white', markeredgewidth=2)
        ax_a.axhline(y=theory_data['eta_theory_static']*100, color='gray',
                     linestyle='--', linewidth=1.5, label='Static Kerr bound')
        ax_a.set_xlabel(r'Mass-loss rate $\\alpha = -\\dot{m}$', fontsize=12)
        ax_a.set_ylabel(r'Efficiency $\\eta$ [%]', fontsize=12)
        ax_a.set_title(r'(a) Efficiency vs. Mass-Loss Rate ($a/M=0.9$)', fontsize=12, fontweight='bold')
        ax_a.legend(fontsize=10)
        ax_a.grid(True, alpha=0.3)

    # Panel B: Efficiency vs. spin
    ax_b = fig.add_subplot(gs[0, 1])
    if spin_static and spin_dynamic:
        a_s = [d['a'] for d in spin_static]
        eta_s = [d['eta'] * 100 for d in spin_static]
        a_d = [d['a'] for d in spin_dynamic]
        eta_d = [d['eta'] * 100 for d in spin_dynamic]
        a_theory = np.linspace(0.3, 0.99, 100)
        eta_theory = [max_penrose_efficiency_static_kerr(M0, ai) * 100 for ai in a_theory]
        ax_b.plot(a_theory, eta_theory, 'k--', linewidth=1.5, label='Theoretical max (static)')
        ax_b.plot(a_s, eta_s, 's-', color='#1f77b4', linewidth=2, markersize=7,
                  label='Numerical (static)', markerfacecolor='white', markeredgewidth=2)
        ax_b.plot(a_d, eta_d, 'D-', color='#ff7f0e', linewidth=2, markersize=7,
                  label='Numerical (dynamic)', markerfacecolor='white', markeredgewidth=2)
        ax_b.set_xlabel(r'Spin parameter $a/M$', fontsize=12)
        ax_b.set_ylabel(r'Efficiency $\\eta$ [%]', fontsize=12)
        ax_b.set_title(r'(b) Efficiency vs. Spin ($\\alpha=5\\times10^{-4}$)', fontsize=12, fontweight='bold')
        ax_b.legend(fontsize=10)
        ax_b.grid(True, alpha=0.3)

    # Panel C: Time-resolved
    ax_c = fig.add_subplot(gs[1, 0])
    if time_data:
        ts = [d['t'] for d in time_data]
        etas_t = [d['eta'] * 100 for d in time_data]
        gaps = [d['gap'] for d in time_data]
        ax_c.plot(ts, etas_t, 'o-', color='#2ca02c', linewidth=2, markersize=5)
        ax_c.set_xlabel(r'Time $t$ [$M$]', fontsize=12)
        ax_c.set_ylabel(r'Efficiency $\\eta$ [%]', fontsize=12)
        ax_c.set_title(r'(c) Time-Resolved Efficiency (Dynamic)', fontsize=12, fontweight='bold')
        ax_c.grid(True, alpha=0.3)
        ax_c2 = ax_c.twinx()
        ax_c2.plot(ts, gaps, ':', color='#d62728', linewidth=1.5, alpha=0.7)
        ax_c2.set_ylabel(r'Horizon gap $r - r_+$ [$M$]', color='#d62728', fontsize=11)
        ax_c2.tick_params(axis='y', labelcolor='#d62728')

    # Panel D: Convergence
    ax_d = fig.add_subplot(gs[1, 1])
    if conv_n:
        ns = [d['n_points'] for d in conv_n]
        etas_n = [d['eta'] * 100 for d in conv_n]
        ax_d.plot(ns, etas_n, 's-', color='#9467bd', linewidth=2, markersize=8,
                  markerfacecolor='white', markeredgewidth=2)
        ax_d.set_xlabel(r'Integration points $N_{\\rm points}$', fontsize=12)
        ax_d.set_ylabel(r'Efficiency $\\eta$ [%]', fontsize=12)
        ax_d.set_title(r'(d) Convergence: $\\eta$ vs. Resolution', fontsize=12, fontweight='bold')
        ax_d.grid(True, alpha=0.3)

    # Panel E: Theory comparison
    ax_e = fig.add_subplot(gs[2, 0])
    labels = ['Static\n(theory)', 'Static\n(numerical)', 'Dynamic\n(numerical)']
    vals = [
        theory_data['eta_theory_static'] * 100,
        theory_data['eta_numerical_static'] * 100 if theory_data['eta_numerical_static'] else 0,
        theory_data['eta_numerical_dynamic'] * 100 if theory_data['eta_numerical_dynamic'] else 0,
    ]
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e']
    bars = ax_e.bar(labels, vals, color=colors, edgecolor='black', linewidth=1.2)
    ax_e.set_ylabel(r'Efficiency $\\eta$ [%]', fontsize=12)
    ax_e.set_title(r'(e) Numerical vs. Theoretical Bound', fontsize=12, fontweight='bold')
    ax_e.grid(True, axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel F: Geometry sketch
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.set_xlim(-3, 3)
    ax_f.set_ylim(-3, 3)
    theta = np.linspace(0, 2*np.pi, 200)
    r_s = 2.0 * M0
    r_p = M0 + np.sqrt(M0**2 - A_DEFAULT**2)
    ax_f.fill(r_s*np.cos(theta), r_s*np.sin(theta), alpha=0.15, color='#ff7f0e', label='Ergosphere (static)')
    ax_f.fill(r_p*np.cos(theta), r_p*np.sin(theta), alpha=0.3, color='black')
    m_dyn = max(M_FLOOR, M0 - ALPHA_DEFAULT * 50)
    r_s_d = 2.0 * m_dyn
    r_p_d = m_dyn + np.sqrt(max(m_dyn**2 - A_DEFAULT**2, 0.0))
    ax_f.plot(r_s_d*np.cos(theta), r_s_d*np.sin(theta), '--', color='#ff7f0e', linewidth=1.5, label='Ergosphere (dynamic, t=50M)')
    ax_f.fill(r_p_d*np.cos(theta), r_p_d*np.sin(theta), alpha=0.15, color='gray')
    ax_f.set_aspect('equal')
    ax_f.set_xlabel(r'$x$ [$M$]', fontsize=12)
    ax_f.set_ylabel(r'$y$ [$M$]', fontsize=12)
    ax_f.set_title(r'(f) Geometry: Static vs. Shrinking Ergosphere', fontsize=12, fontweight='bold')
    ax_f.legend(fontsize=9, loc='upper right')
    ax_f.grid(True, alpha=0.3)

    fig.suptitle(
        r'Optimal Energy Extraction from the Dynamic Ergosphere of Kerr-Vaidya Black Holes',
        fontsize=16, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(OUT_PDF, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {OUT_PNG}")
    print(f"  Saved: {OUT_PDF}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Step 05: Comprehensive Parametric Study & Publication Figure")
    print(f"Resolution mode: {RES_MODE}")
    print("=" * 70)
    alpha_data = section_alpha_sweep()
    spin_static, spin_dynamic = section_spin_sweep()
    time_data = section_time_resolved()
    conv_results = section_convergence()
    conv_n, conv_grid = conv_results if conv_results else (None, None)
    theory_data = section_theory()
    plot_publication_figure(alpha_data, spin_static, spin_dynamic, time_data, conv_n, conv_grid, theory_data)
    print("\n" + "=" * 70)
    print("Step 05 complete. All data saved to data/")
    print("=" * 70)