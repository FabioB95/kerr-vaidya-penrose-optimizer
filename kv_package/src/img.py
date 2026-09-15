"""figures_final.py -- run in kv_package/src/ after add_alpha_crit_099.py"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Robust path: works no matter where you launch the script from
SCRIPT_DIR = Path(__file__).resolve().parent
json_path = SCRIPT_DIR / "sweep_results_v3.json"
if not json_path.exists():
    json_path = SCRIPT_DIR.parent / "sweep_results_v3.json"
if not json_path.exists():
    raise FileNotFoundError(
        f"Could not find sweep_results_v3.json.\n"
        f"Tried:\n  {SCRIPT_DIR / 'sweep_results_v3.json'}\n  {SCRIPT_DIR.parent / 'sweep_results_v3.json'}"
    )

with open(json_path) as f:
    D = json.load(f)

def r_plus(m, a):
    return m + np.sqrt(max(m**2 - a**2, 0.0))
def r_s(m):
    return 2 * m

STATIC_SPINS = [0.80, 0.90, 0.95, 0.98, 0.99]

# --- Kerr geometry ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
spins_geom = [0.0, 0.5, 0.9, 0.998]
theta = np.linspace(0, 2*np.pi, 400)
for ax, a in zip(axes, spins_geom):
    m = 1.0
    rp, rs = r_plus(m, a), r_s(m)
    ax.fill(rs*np.cos(theta), rs*np.sin(theta), color='orange', alpha=0.35, label='Ergosphere')
    ax.fill(rp*np.cos(theta), rp*np.sin(theta), color='black', alpha=0.75, label='Horizon')
    ax.set_title(f"a/M = {a}")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.text(0.05, 0.05, f"$r_s-r_+$={rs-rp:.3f}M", transform=ax.transAxes, fontsize=9)
axes[0].legend(loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig("fig_kerr_geometry.pdf")
print("Saved fig_kerr_geometry.pdf")

# --- alpha sweep ---
alphas, dEs, etas = [], [], []
for key, val in D.items():
    if key.startswith("alphasweep_a0.9_alpha") and val:
        alphas.append(float(key.split("alpha")[-1]))
        dEs.append(val["dE"]); etas.append(val["eta"]*100)
order = np.argsort(alphas)
alphas, dEs, etas = np.array(alphas)[order], np.array(dEs)[order], np.array(etas)[order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(alphas, dEs, 'o-', color='steelblue')
ax1.set_xlabel(r"Mass-loss rate $\alpha$"); ax1.set_ylabel(r"$\Delta E = E_3-E_1$  [M]")
ax1.set_title("Physically bounded energy gain vs. $\\alpha$ (a/M=0.9)")
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))   # <-- fix overlapping labels

ax2.plot(alphas, etas, 's-', color='darkorange')
ax2.set_xlabel(r"Mass-loss rate $\alpha$"); ax2.set_ylabel(r"$\eta$ [%]")
ax2.set_title("Efficiency vs. $\\alpha$ (a/M=0.9)")
ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))   # <-- fix overlapping labels

fig.tight_layout()
fig.savefig("fig_alpha_sweep.pdf")
print("Saved fig_alpha_sweep.pdf")

# --- spin dependence ---
static_dE = [D[f"static_a{a}"]["dE"] for a in STATIC_SPINS]
static_eta = [D[f"static_a{a}"]["eta"]*100 for a in STATIC_SPINS]
dynamic_dE = [D[f"dynamic_a{a}_alpha5e-4"]["dE"] for a in STATIC_SPINS]
dynamic_eta = [D[f"dynamic_a{a}_alpha5e-4"]["eta"]*100 for a in STATIC_SPINS]
dynamic_a_over_m = [a / D[f"dynamic_a{a}_alpha5e-4"]["m_split"] for a in STATIC_SPINS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
ax1.plot(STATIC_SPINS, static_dE, 'o-', label='Static (alpha=0)', color='steelblue')
ax1.plot(STATIC_SPINS, dynamic_dE, 's--', label=r'Dynamic ($\alpha=5\times10^{-4}$)', color='darkorange')
ax1.set_xlabel("a/M (nominal)"); ax1.set_ylabel(r"$\Delta E$ [M]")
ax1.set_title("Physically bounded energy gain vs. nominal spin")
ax1.legend()
ax2.plot(STATIC_SPINS, static_eta, 'o-', label='Static', color='steelblue')
ax2.plot(STATIC_SPINS, dynamic_eta, 's--', label='Dynamic', color='darkorange')
for a, aom in zip(STATIC_SPINS, dynamic_a_over_m):
    ax2.annotate(f"a/m={aom:.3f}", (a, D[f'dynamic_a{a}_alpha5e-4']['eta']*100),
                 textcoords="offset points", xytext=(0, 8), fontsize=7, color='darkorange')
ax2.set_xlabel("a/M (nominal)"); ax2.set_ylabel(r"$\eta$ [%]")
ax2.set_title("Efficiency vs. nominal spin (annotated: actual a/m at split time)")
ax2.legend()
fig.tight_layout()
fig.savefig("fig_spin_dependence.pdf")
print("Saved fig_spin_dependence.pdf")

# --- convergence ---
ns, dEs_c, etas_c = [], [], []
for key, val in D.items():
    if key.startswith("convergence_n") and val:
        ns.append(int(key.replace("convergence_n", "")))
        dEs_c.append(val["dE"]); etas_c.append(val["eta"]*100)
order = np.argsort(ns)
ns, dEs_c, etas_c = np.array(ns)[order], np.array(dEs_c)[order], np.array(etas_c)[order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(ns, dEs_c, 'o-', color='seagreen')
ax1.set_xlabel("Grid resolution (n)"); ax1.set_ylabel(r"$\Delta E$ [M]")
ax1.set_title(r"$\Delta E$ convergence (a/M=0.9, static)")
ax1.set_ylim(min(dEs_c)*0.95, max(dEs_c)*1.05)
ax2.plot(ns, etas_c, 's-', color='purple')
ax2.set_xlabel("Grid resolution (n)"); ax2.set_ylabel(r"$\eta$ [%]")
ax2.set_title(r"$\eta$ convergence")
fig.tight_layout()
fig.savefig("fig_convergence.pdf")
print("Saved fig_convergence.pdf")

# --- phase boundary (handles 2 or 3 points gracefully) ---
crit_spins = [a for a in [0.90, 0.95, 0.99] if D.get(f"alpha_crit_a{a}") is not None]
crit_vals = [D[f"alpha_crit_a{a}"] for a in crit_spins]
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(crit_spins, crit_vals, 'o-', color='crimson', markersize=10)
for a, c in zip(crit_spins, crit_vals):
    ax.annotate(f"{c:.4f}", (a, c), textcoords="offset points", xytext=(10, 0))
ax.set_xlabel("a/M"); ax.set_ylabel(r"$\alpha_{\rm crit}$")
ax.set_title(r"Phase boundary: $\alpha_{\rm crit}$ vs spin")
fig.tight_layout()
fig.savefig("fig_phase_boundary.pdf")
print(f"Saved fig_phase_boundary.pdf ({len(crit_spins)} points: {crit_spins})")

# --- Table 3 ---
print("\n--- Table 3 (corrected) ---")
print(f"{'a/M':>6} | {'dE_static':>10} | {'eta_static':>11} | {'dE_dynamic':>11} | {'eta_dynamic':>12} | {'a/m_split(dyn)':>15}")
for a in STATIC_SPINS:
    s, d = D[f"static_a{a}"], D[f"dynamic_a{a}_alpha5e-4"]
    print(f"{a:>6} | {s['dE']:>10.4f} | {s['eta']*100:>10.2f}% | {d['dE']:>11.4f} | "
          f"{d['eta']*100:>11.2f}% | {a/d['m_split']:>15.4f}")

plt.show()