#!/usr/bin/env python3
"""
Step 04 Replot: Tight zoom on black hole region
"""
import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import matplotlib.pyplot as plt
from kerr_vaidya_metric import (
    KerrVaidyaParams, integrate_geodesic, effective_potential, penrose_efficiency,
    plot_ergosphere_and_horizon,
)

M0, A, ALPHA, M_FLOOR = 1.0, 0.9, 0.0005, 0.95
R0, PHI0, R_ESCAPE = 6.0, 0.0, 5.0

def mfs(t):
    t = np.asarray(t)
    return np.full_like(t, M0, dtype=float)
def mds(t):
    t = np.asarray(t)
    return np.zeros_like(t, dtype=float)
def mfd(t):
    t = np.asarray(t)
    return np.maximum(M_FLOOR, M0 - ALPHA * t)
def mdd(t):
    t = np.asarray(t)
    m = M0 - ALPHA * t
    return np.where(m > M_FLOOR, -ALPHA, 0.0)

PS = KerrVaidyaParams(a=A, m_func=mfs, m_dot_func=mds)
PD = KerrVaidyaParams(a=A, m_func=mfd, m_dot_func=mdd)

# Static ICs
E1s, L1s, pr0s = 0.6731, 1.8923, -0.7318
# Dynamic ICs
E1d, L1d, pr0d = 0.7654, 1.8923, -0.1864

# Static split
rs, phis, ts = 1.7593, 0.0, 24.81
E2s, L2s = -0.7370, 0.1000
E3s, L3s, pr3s = 1.4502, 1.8895, 10.00

# Dynamic split
rd, phid, td = 1.6522, 0.0, 59.95
E2d, L2d = -0.3459, 0.1000
E3d, L3d, pr3d = 1.2486, 2.1421, 10.00

def run(label, params, E1, L1, pr0, rsplit, phisplit, tsplit, E2, L2, E3, L3, pr3):
    print(f"[{label}] Integrating...")
    tin = integrate_geodesic(r0=R0, phi0=PHI0, E=E1, L=L1, p_r0=pr0,
                             tau_max=300.0, params=params, n_points=1500)
    # find deepest
    rsv = 2.0 * params.m_func(tin['t'])
    rpm = params.m_func(tin['t']) + np.sqrt(np.maximum(params.m_func(tin['t'])**2 - params.a**2, 0.0))
    inside = (tin['r'] <= rsv) & (tin['r'] >= rpm + 0.30)
    idx = np.where(inside)[0][np.argmin(tin['r'][inside])] if np.any(inside) else len(tin['r'])//2

    t2 = integrate_geodesic(r0=rsplit, phi0=phisplit, E=E2, L=L2, p_r0=-0.05,
                            tau_max=100.0, params=params, n_points=800)
    t3 = integrate_geodesic(r0=rsplit, phi0=phisplit, E=E3, L=L3, p_r0=pr3,
                            tau_max=600.0, params=params, n_points=2500)
    eta = penrose_efficiency(E1, E2, E3)
    return tin, idx, t2, t3, eta, params.m_func(np.array([tsplit]))[0]

res_s = run('Static', PS, E1s, L1s, pr0s, rs, phis, ts, E2s, L2s, E3s, L3s, pr3s)
res_d = run('Dynamic', PD, E1d, L1d, pr0d, rd, phid, td, E2d, L2d, E3d, L3d, pr3d)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, label, res, params in zip(axes, ['Static', 'Dynamic'], [res_s, res_d], [PS, PD]):
    tin, idx, t2, t3, eta, ms = res

    # Geometry
    plot_ergosphere_and_horizon(m=ms, a=A, ax=ax, label=False)
    if label == 'Dynamic':
        theta = np.linspace(0, 2*np.pi, 200)
        m_early = max(M_FLOOR, ms + 0.03)
        if m_early > M_FLOOR:
            ax.plot(2.0*m_early*np.cos(theta), 2.0*m_early*np.sin(theta),
                    '--', color='#ff7f0e', linewidth=1.2, alpha=0.6)

    x_pre = tin['r'][:idx+1] * np.cos(tin['phi'][:idx+1])
    y_pre = tin['r'][:idx+1] * np.sin(tin['phi'][:idx+1])
    x2 = t2['r'] * np.cos(t2['phi'])
    y2 = t2['r'] * np.sin(t2['phi'])
    x3 = t3['r'] * np.cos(t3['phi'])
    y3 = t3['r'] * np.sin(t3['phi'])

    ax.plot(x_pre, y_pre, color='#1f77b4', linewidth=2.2, label='Incoming', zorder=3)
    ax.plot(x2, y2, color='#d62728', linewidth=1.8, linestyle='--', label='Fragment 2', zorder=3)
    ax.plot(x3, y3, color='#2ca02c', linewidth=1.8, linestyle='-.', label='Fragment 3', zorder=3)

    xs = tin['r'][idx] * np.cos(tin['phi'][idx])
    ys = tin['r'][idx] * np.sin(tin['phi'][idx])
    ax.scatter([xs], [ys], color='black', s=100, zorder=5, marker='X',
               edgecolors='white', linewidths=1.5)
    ax.scatter([R0*np.cos(PHI0)], [R0*np.sin(PHI0)], color='#1f77b4',
               s=50, zorder=5, marker='o', edgecolors='white', linewidths=1)

    # KEY FIX: tight limits, ignoring the long escape tail
    # Compute max range ONLY from pre-split + fragment 2 (near-BH stuff)
    near_x = np.concatenate([x_pre, x2])
    near_y = np.concatenate([y_pre, y2])
    margin = 0.8
    max_r = max(np.max(np.abs(near_x)), np.max(np.abs(near_y))) + margin
    # Cap at 7M so the figure stays focused
    max_r = min(max_r, 7.0)
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)

    ax.set_aspect('equal')
    ax.set_xlabel(r'$x \; [M]$', fontsize=11)
    ax.set_ylabel(r'$y \; [M]$', fontsize=11)

    if label == 'Static':
        title, ann = 'Static Kerr', rf'$\eta = {eta*100:.1f}\%$' + '\n' + rf'$m = {ms:.3f}M$'
    else:
        title = 'Dynamic Kerr--Vaidya'
        ann = rf'$\eta = {eta*100:.1f}\%$' + '\n' + rf'$m = {ms:.3f}M$' + '\n' + rf'$\dot{{m}} = -{ALPHA:.4f}$'

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=10,
            va='top', ha='left', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.9))
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(project_root, 'figures', 'step04_optimizer_comparison.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
fig.savefig(os.path.join(project_root, 'figures', 'step04_optimizer_comparison.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print("Saved: step04_optimizer_comparison.pdf/.png")