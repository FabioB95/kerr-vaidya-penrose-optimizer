#!/usr/bin/env python3
"""
================================================================================
FILE: src/step01_visualize_kerr_geometry.py
================================================================================

Step 01: Visualize Kerr Geometry for Different Spin Parameters
==============================================================

This script generates a 2x2 grid showing the ergosphere and event horizon
for four spin parameters: a/M = 0.0, 0.5, 0.9, 0.998.

Outputs:
    figures/step01_kerr_geometry_spins.png
    figures/step01_kerr_geometry_spins.pdf

Run this in your VS Code terminal:
    cd /path/to/kerr_vaidya_ergosphere
    python src/step01_visualize_kerr_geometry.py
================================================================================
"""

import sys
import os

# Add src/ to Python path so we can import the metric module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import matplotlib.pyplot as plt
from kerr_vaidya_metric import (
    event_horizon_radius,
    ergosphere_radius,
)

# ==============================================================================
# Configuration
# ==============================================================================

M = 1.0                     # Black hole mass (in units of M)
SPINS = [0.0, 0.5, 0.9, 0.998]
TITLES = [
    r'Schwarzschild  $a/M = 0$',
    r'$a/M = 0.5$',
    r'$a/M = 0.9$',
    r'Near-extremal  $a/M = 0.998$',
]
FIGSIZE = (12, 12)
XLIM = (-5, 5)
YLIM = (-5, 5)
N_THETA = 500               # Angular resolution for boundary curves

# Output paths (relative to project root)
OUT_PNG = os.path.join(project_root, 'figures', 'step01_kerr_geometry_spins.png')
OUT_PDF = os.path.join(project_root, 'figures', 'step01_kerr_geometry_spins.pdf')
DPI = 300

# ==============================================================================
# Plotting
# ==============================================================================

def plot_kerr_geometry(ax, m, a, title):
    """Plot ergosphere and horizon on the given axes."""
    theta = np.linspace(0, 2 * np.pi, N_THETA)

    # Ergosphere boundary
    r_s_plus, _ = ergosphere_radius(m, a, theta)
    x_ergo = r_s_plus * np.cos(theta)
    y_ergo = r_s_plus * np.sin(theta)

    # Event horizon
    r_plus, _ = event_horizon_radius(m, a)
    x_horizon = r_plus * np.cos(theta)
    y_horizon = r_plus * np.sin(theta)

    # Fill regions
    ax.fill(x_ergo, y_ergo, alpha=0.20, color='#FF6B00', label='Ergosphere')
    ax.fill(x_horizon, y_horizon, alpha=0.40, color='#1A1A1A', label='Horizon')

    # Boundary curves
    ax.plot(x_ergo, y_ergo, color='#FF6B00', linewidth=1.5, linestyle='-')
    ax.plot(x_horizon, y_horizon, color='#1A1A1A', linewidth=2.0, linestyle='-')

    # Reference circles
    ax.axhline(0, color='gray', linewidth=0.3, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.3, alpha=0.5)

    # Annotations
    ax.text(0.05, 0.95, title, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    # Horizon radius annotation
    if not np.isnan(r_plus):
        ax.text(0.05, 0.05, f'$r_+ = {r_plus:.3f}M$',
                transform=ax.transAxes, fontsize=10, va='bottom', ha='left',
                color='#1A1A1A', fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xlabel(r'$x \; [M]$', fontsize=11)
    ax.set_ylabel(r'$y \; [M]$', fontsize=11)

    # Only show legend on first panel
    if a == SPINS[0]:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(
        r'Kerr Geometry: Ergosphere and Event Horizon for Different Spin Parameters',
        fontsize=15, fontweight='bold', y=0.98
    )

    for ax, a_spin, title in zip(axes.flat, SPINS, TITLES):
        plot_kerr_geometry(ax, M, a_spin, title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save outputs
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUT_PDF, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    plt.close(fig)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
    print("Step 01 complete.")