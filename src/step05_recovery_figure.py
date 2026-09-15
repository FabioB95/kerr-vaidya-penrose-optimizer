#!/usr/bin/env python3
"""
================================================================================
Step 05 Recovery: Generate Figure from Saved JSON Data
================================================================================

Run this after step05_comprehensive_study.py crashes during figure saving.
All computed data is loaded from data/section_*.json — no re-computation needed.

Run:
    python src/step05_recovery_figure.py
================================================================================
"""

import sys
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from kerr_vaidya_metric import max_penrose_efficiency_static_kerr

# ==============================================================================
# Config
# ==============================================================================

M0 = 1.0
A_DEFAULT = 0.9
ALPHA_DEFAULT = 0.0005
M_FLOOR = 0.95

DATA_DIR = os.path.join(project_root, 'data')
OUT_DIR = os.path.join(project_root, 'figures')
OUT_PNG = os.path.join(OUT_DIR, 'step05_comprehensive_study.png')
OUT_PDF = os.path.join(OUT_DIR, 'step05_comprehensive_study.pdf')
DPI = 300

# ==============================================================================
# Load data
# ==============================================================================

def load_json(name):
    path = os.path.join(DATA_DIR, f'section_{name}.json')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    with open(path, 'r') as f:
        return json.load(f)

alpha_data = load_json('alpha')
spin_data = load_json('spin')
time_data = load_json('time')
conv_data = load_json('conv')
theory_data = load_json('theory')

spin_static = spin_data.get('static', []) if spin_data else []
spin_dynamic = spin_data.get('dynamic', []) if spin_data else []
conv_n = conv_data.get('n_points_sweep', []) if conv_data else []
conv_grid = conv_data.get('grid_sweep', []) if conv_data else []

print(f"Loaded: alpha={len(alpha_data) if alpha_data else 0} points")
print(f"Loaded: spin static={len(spin_static)}, dynamic={len(spin_dynamic)}")
print(f"Loaded: time={len(time_data) if time_data else 0} points")
print(f"Loaded: conv n={len(conv_n)}, grid={len(conv_grid)}")
print(f"Loaded: theory={theory_data is not None}")

# ==============================================================================
# Build figure
# ==============================================================================

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30)

# --- Panel A: Efficiency vs. alpha ---
ax_a = fig.add_subplot(gs[0, 0])
if alpha_data:
    alphas = [d['alpha'] for d in alpha_data]
    etas = [d['eta'] * 100 for d in alpha_data]
    ax_a.plot(alphas, etas, 'o-', color='#1f77b4', linewidth=2, markersize=8,
              markerfacecolor='white', markeredgewidth=2)
    if theory_data and theory_data.get('eta_theory_static'):
        ax_a.axhline(y=theory_data['eta_theory_static']*100, color='gray',
                     linestyle='--', linewidth=1.5, label='Static Kerr bound')
    ax_a.set_xlabel('Mass-loss rate ' + r'$\alpha = -\dot{m}$', fontsize=12)
    ax_a.set_ylabel('Efficiency ' + r'$\eta$ [%]', fontsize=12)
    ax_a.set_title('(a) Efficiency vs. Mass-Loss Rate ($a/M=0.9$)', fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=10)
    ax_a.grid(True, alpha=0.3)

# --- Panel B: Efficiency vs. spin ---
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
    ax_b.set_xlabel('Spin parameter $a/M$', fontsize=12)
    ax_b.set_ylabel('Efficiency ' + r'$\eta$ [%]', fontsize=12)
    ax_b.set_title('(b) Efficiency vs. Spin ($\\alpha=5\\times10^{-4}$)', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.3)

# --- Panel C: Time-resolved ---
ax_c = fig.add_subplot(gs[1, 0])
if time_data:
    ts = [d['t'] for d in time_data]
    etas_t = [d['eta'] * 100 for d in time_data]
    gaps = [d['gap'] for d in time_data]
    ax_c.plot(ts, etas_t, 'o-', color='#2ca02c', linewidth=2, markersize=5)
    ax_c.set_xlabel('Time $t$ [$M$]', fontsize=12)
    ax_c.set_ylabel('Efficiency ' + r'$\eta$ [%]', fontsize=12)
    ax_c.set_title('(c) Time-Resolved Efficiency (Dynamic)', fontsize=12, fontweight='bold')
    ax_c.grid(True, alpha=0.3)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(ts, gaps, ':', color='#d62728', linewidth=1.5, alpha=0.7)
    ax_c2.set_ylabel('Horizon gap $r - r_+$ [$M$]', color='#d62728', fontsize=11)
    ax_c2.tick_params(axis='y', labelcolor='#d62728')

# --- Panel D: Convergence ---
ax_d = fig.add_subplot(gs[1, 1])
if conv_n:
    ns = [d['n_points'] for d in conv_n]
    etas_n = [d['eta'] * 100 for d in conv_n]
    ax_d.plot(ns, etas_n, 's-', color='#9467bd', linewidth=2, markersize=8,
              markerfacecolor='white', markeredgewidth=2)
    ax_d.set_xlabel('Integration points $N_{\\rm points}$', fontsize=12)
    ax_d.set_ylabel('Efficiency ' + r'$\eta$ [%]', fontsize=12)
    ax_d.set_title('(d) Convergence: ' + r'$\eta$ vs. Resolution', fontsize=12, fontweight='bold')
    ax_d.grid(True, alpha=0.3)

# --- Panel E: Theory comparison ---
ax_e = fig.add_subplot(gs[2, 0])
if theory_data:
    labels = ['Static\n(theory)', 'Static\n(numerical)', 'Dynamic\n(numerical)']
    vals = [
        theory_data.get('eta_theory_static', 0) * 100,
        theory_data.get('eta_numerical_static', 0) * 100 if theory_data.get('eta_numerical_static') else 0,
        theory_data.get('eta_numerical_dynamic', 0) * 100 if theory_data.get('eta_numerical_dynamic') else 0,
    ]
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e']
    bars = ax_e.bar(labels, vals, color=colors, edgecolor='black', linewidth=1.2)
    ax_e.set_ylabel('Efficiency ' + r'$\eta$ [%]', fontsize=12)
    ax_e.set_title('(e) Numerical vs. Theoretical Bound', fontsize=12, fontweight='bold')
    ax_e.grid(True, axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# --- Panel F: Geometry sketch ---
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
ax_f.set_xlabel('$x$ [$M$]', fontsize=12)
ax_f.set_ylabel('$y$ [$M$]', fontsize=12)
ax_f.set_title('(f) Geometry: Static vs. Shrinking Ergosphere', fontsize=12, fontweight='bold')
ax_f.legend(fontsize=9, loc='upper right')
ax_f.grid(True, alpha=0.3)

fig.suptitle(
    'Optimal Energy Extraction from the Dynamic Ergosphere of Kerr-Vaidya Black Holes',
    fontsize=16, fontweight='bold', y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
fig.savefig(OUT_PDF, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)

print(f"\nSaved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
print("Recovery complete!")