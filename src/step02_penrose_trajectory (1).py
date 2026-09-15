#!/usr/bin/env python3
"""
Step 02 (FIXED): Schematic Penrose trajectory for static Kerr a/M=0.9 (Fig. 2)

Bug fixed: the incoming trajectory used to be generated from an oscillating
r(phi) formula and then truncated at a FIXED array index ([:180]) regardless
of where r actually was at that point. Numerically, r[180] = 2.91M, which is
OUTSIDE the ergosphere (r_s = 2.0M) -- so the split point (and both fragments,
anchored to it) were plotted in empty space, disconnected from the black hole.

Fix: the trajectory is now built by construction to land EXACTLY on the
intended split radius (just inside r_s, comfortably above r_+) at its last
point -- see _penrose_common.incoming_trajectory.
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

fig, ax = plt.subplots(figsize=(6.5, 6.5))
M, a = 1.0, 0.7  # matches the current caption (differentiated from Fig. 2's a/M=0.9)
r_plus = M + np.sqrt(M**2 - a**2)
r_s = 2.0 * M

E1, E2, E3 = 0.95, -0.28, 1.23  # eta = -E2/E1 = 29.5%, matches caption exactly
ETA_PCT = 29.5

# Split point: comfortably INSIDE the ergosphere (matches Fig. captions: r_split ~ 1.6M)
r_split = 1.80  # a bit deeper than r_plus=1.714 since the annulus is thinner at a/M=0.7
phi_split = 2.3

r_in, phi_in = incoming_trajectory(r_start=6.0, r_split=r_split, phi_total=phi_split)
x_in, y_in = to_xy(r_in, phi_in)

x_s, y_s = to_xy(r_split, phi_split)

# Fragment 2: plunges well inside r_plus so it visually disappears behind the horizon
r2, phi2 = fragment_arc(r_split, phi_split, r_end=r_plus * 0.3, phi_extra=0.45, n=60, power=3.0)
x2, y2 = to_xy(r2, phi2)

# Fragment 3: escapes well past the plot margin
r3, phi3 = fragment_arc(r_split, phi_split, r_end=8.0, phi_extra=-1.3, curve=0.3, n=90)
x3, y3 = to_xy(r3, phi3)

# Ergosphere and horizon
theta = np.linspace(0, 2 * np.pi, 400)
x_ergo, y_ergo = to_xy(r_s, theta)
ax.fill(x_ergo, y_ergo, color='orange', alpha=0.2, zorder=1)
ax.plot(x_ergo, y_ergo, color='orange', linewidth=1.0, zorder=1.5, label=r'$r_{\rm s}$')
x_h, y_h = to_xy(r_plus, theta)
ax.fill(x_h, y_h, color='black', zorder=3)
ax.plot(x_h, y_h, color='black', linewidth=1.5, zorder=3.5, label=r'$r_+$')

# Trajectories (fragment 2 drawn BELOW the horizon fill -> plunges "into" it)
ax.plot(x_in, y_in, color='#1f77b4', linewidth=2.2, zorder=4,
        label=rf'Incoming ($E_1={E1:.2f}$)')
ax.plot(x2, y2, color='#d62728', linewidth=2.2, linestyle='--', zorder=2,
        label=rf'Fragment 2 ($E_2={E2:.2f}$)')
ax.plot(x3, y3, color='#2ca02c', linewidth=2.2, linestyle='-.', zorder=4,
        label=rf'Fragment 3 ($E_3={E3:.2f}$)')
ax.scatter([x_s], [y_s], color='black', s=130, zorder=5, marker='X',
           edgecolors='white', linewidths=2)
ax.scatter([6.0], [0.0], color='#1f77b4', s=80, zorder=5, marker='o',
           edgecolors='white', linewidths=1.5)

ax.set_aspect('equal')
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.set_xlabel(r'$x/M$', fontsize=12)
ax.set_ylabel(r'$y/M$', fontsize=12)
ax.set_title(rf'Static Kerr, $a/M={a}$', fontsize=12)
ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
ax.text(0.02, 0.98, rf'$\eta = {ETA_PCT}\%$', transform=ax.transAxes, fontsize=11,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

plt.tight_layout()
fig.savefig(f'{OUTDIR}/step02_penrose_trajectory.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUTDIR}/step02_penrose_trajectory.pdf', bbox_inches='tight')
plt.close()
print('Saved step02 (fixed)')
