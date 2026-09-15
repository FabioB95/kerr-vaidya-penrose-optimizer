"""Shared helper for schematic Penrose trajectories with guaranteed endpoints."""
import numpy as np

def incoming_trajectory(r_start, r_split, phi_total, wiggle=0.35, n=220):
    """Radial trajectory from r_start to EXACTLY r_split at phi_total, by construction."""
    t = np.linspace(0.0, 1.0, n)
    base = r_start + (r_split - r_start) * t
    bump = wiggle * np.sin(2.2 * np.pi * t) * (t * (1 - t)) * 4.0
    r = base + bump
    phi = phi_total * t
    return r, phi

def fragment_arc(r_split, phi_split, r_end, phi_extra, curve=0.0, n=70, power=1.0):
    """Fragment trajectory starting EXACTLY at (r_split, phi_split).

    power > 1 keeps the curve closer to r_split for longer (slow start, then
    a quick dive) -- useful for the plunging fragment so a visible stub of
    line remains before it disappears behind the horizon, instead of being
    hidden immediately under the split-point marker.
    """
    t = np.linspace(0.0, 1.0, n)
    r = r_split + (r_end - r_split) * t**power + curve * np.sin(np.pi * t) * (t * (1 - t)) * 4.0
    phi = phi_split + phi_extra * t
    return r, phi

def to_xy(r, phi):
    return r * np.cos(phi), r * np.sin(phi)
