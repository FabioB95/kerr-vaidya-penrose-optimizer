#!/usr/bin/env python3
"""
Step 04 (FIXED): Optimized Penrose trajectories (Fig. 4)

Bug fixed: the split marker (X) correctly used the real r_min from the
numerical log, but the "Incoming" curve was built from a separate formula
truncated at a fixed index that never actually reached r_min (it stopped at
r=2.51M for the static case, r=3.23M dynamic) -- leaving a visible gap
between the end of the blue curve and the X marker, and between the X marker
and where Fragment 2 / Fragment 3 visually started.

Fix: incoming trajectory now built to terminate EXACTLY at r_min by
construction (see _penrose_common.incoming_trajectory), and fragments now
start exactly at that same point.

Data from actual numerical logs:
  Static:  E1=0.7132, eta=103.35%, t_split=26.58M, r_min=1.7376M
  Dynamic: E1=0.9026, eta=38.32%,  t_split=58.63M, r_min=1.6398M
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from _penrose_common import incoming_trajectory, fragment_arc, to_xy

OUTDIR = '/home/claude/fixed/out'
os.makedirs(OUTDIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
M, a = 1.0, 0.9
ALPHA = 0.0005

static_data = {
    'E1': 0.7132, 'eta': 103.35, 'E2': -0.7370, 'E3': 1.4502,
    'p_r3': 10.0, 'm_split': 1.0, 'r_min': 1.7376,
}
dynamic_data = {
    'E1': 0.9026, 'eta': 38.32, 'E2': -0.3459, 'E3': 1.2486,
    'p_r3': 10.0, 'm_split': 0.9707, 'r_min': 1.6398,
}

for ax, label in zip(axes, ['Static', 'Dynamic']):
    is_dyn = (label == 'Dynamic')
    data = dynamic_data if is_dyn else static_data
    m = data['m_split']
    r_plus = m + np.sqrt(max(m**2 - a**2, 0.0))
    r_s = 2.0 * m
    r_split = data['r_min']
    phi_split = 2.3

    theta = np.linspace(0, 2 * np.pi, 400)
    x_ergo, y_ergo = to_xy(r_s, theta)
    ax.fill(x_ergo, y_ergo, color='orange', alpha=0.2, zorder=1)
    ax.plot(x_ergo, y_ergo, color='orange', linewidth=1.0, zorder=1.5)
    x_h, y_h = to_xy(r_plus, theta)
    ax.fill(x_h, y_h, color='black', zorder=3)
    ax.plot(x_h, y_h, color='black', linewidth=1.5, zorder=3.5)

    if is_dyn:
        for dt in [20, 50, 100]:
            t_later = 58.63 + dt
            m_later = max(0.95, M - ALPHA * t_later)
            x_l, y_l = to_xy(2.0 * m_later, theta)
            ax.plot(x_l, y_l, color='orange', linewidth=0.7, alpha=0.35, linestyle='--')

    # Incoming trajectory -- now guaranteed to end exactly at (r_split, phi_split)
    r_in, phi_in = incoming_trajectory(r_start=6.0, r_split=r_split, phi_total=phi_split)
    x_in, y_in = to_xy(r_in, phi_in)
    x_sp, y_sp = to_xy(r_split, phi_split)

    r2, phi2 = fragment_arc(r_split, phi_split, r_end=r_plus * 0.3, phi_extra=0.4, n=60, power=3.0)
    x2, y2 = to_xy(r2, phi2)
    r3, phi3 = fragment_arc(r_split, phi_split, r_end=8.5, phi_extra=-1.3, curve=0.3, n=90)
    x3, y3 = to_xy(r3, phi3)

    ax.plot(x_in, y_in, color='#1f77b4', linewidth=2.5, zorder=4,
            label=rf'Incoming ($E_1={data["E1"]:.2f}$)')
    ax.plot(x2, y2, color='#d62728', linewidth=2.2, linestyle='--', zorder=2,
            label=rf'Fragment 2 ($E_2={data["E2"]:.2f}$)')
    ax.plot(x3, y3, color='#2ca02c', linewidth=2.2, linestyle='-.', zorder=4,
            label=rf'Fragment 3 ($E_3={data["E3"]:.2f}$)')
    ax.scatter([x_sp], [y_sp], color='black', s=150, zorder=5, marker='X',
               edgecolors='white', linewidths=2.5)
    ax.scatter([6.0], [0.0], color='#1f77b4', s=100, zorder=5, marker='o',
               edgecolors='white', linewidths=2.0)

    ax.set_aspect('equal')
    margin = 7.0
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_xlabel(r'$x/M$', fontsize=12)
    ax.set_ylabel(r'$y/M$', fontsize=12)
    ax.set_title(f'{label} Case', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

    if is_dyn:
        ann = (f'{label} Kerr-Vaidya\n'
               f'$a/M = {a:.1f}$\n'
               f'$m = {data["m_split"]:.3f}M$\n'
               f'$r_s = {r_s:.3f}M$\n'
               f'$r_+ = {r_plus:.3f}M$\n'
               f'$\\dot{{m}} = -{ALPHA:.4f}$\n'
               f'$p_{{r,3}} = {data["p_r3"]:.1f}$\n'
               f'$\\eta = {data["eta"]:.1f}\\%$\n'
               'ESCAPED')
    else:
        ann = (f'{label} Kerr\n'
               f'$a/M = {a:.1f}$\n'
               f'$m = {data["m_split"]:.3f}M$\n'
               f'$r_s = {r_s:.3f}M$\n'
               f'$r_+ = {r_plus:.3f}M$\n'
               f'$p_{{r,3}} = {data["p_r3"]:.1f}$\n'
               f'$\\eta = {data["eta"]:.1f}\\%$\n'
               'ESCAPED')
    ax.text(0.02, 0.98, ann, transform=ax.transAxes, fontsize=10, va='top', ha='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.95))

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{OUTDIR}/step04_optimizer_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUTDIR}/step04_optimizer_comparison.pdf', bbox_inches='tight')
plt.close()
print('Saved step04 (fixed)')
