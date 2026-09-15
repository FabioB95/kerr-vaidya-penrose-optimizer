"""fig_shrinking_ergosphere.py -- run in kv_package/src/"""
import numpy as np
import matplotlib.pyplot as plt
import geodesics as kv

a = 0.9
m0 = 1.0
alpha = 5e-4
theta = np.linspace(0, 2*np.pi, 300)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# --- left: static reference, single snapshot ---
rp0 = kv.r_plus(m0, a)
rs0 = kv.r_s(m0)
ax1.fill(rs0*np.cos(theta), rs0*np.sin(theta), color='orange', alpha=0.35, label='Ergosphere')
ax1.fill(rp0*np.cos(theta), rp0*np.sin(theta), color='black', alpha=0.85, label='Horizon')
ax1.set_title(f"Static Kerr (a/M={a}, alpha=0)")
ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.5, 2.5)
ax1.set_aspect('equal')
ax1.legend(loc='upper right', fontsize=9)
ax1.text(0.05, 0.05, f"$r_s-r_+$={rs0-rp0:.3f}M (constant)", transform=ax1.transAxes, fontsize=9)

# --- right: dynamic, concentric snapshots at several advanced times ---
v_snapshots = [0, 25, 50, 75, 100]
cmap = plt.cm.Oranges(np.linspace(0.3, 0.9, len(v_snapshots)))
for v, color in zip(v_snapshots, cmap):
    m_v = kv.m_of_v(v, m0, alpha) if hasattr(kv, 'm_of_v') else max(m0 - alpha*v, 0.0)
    rp_v = kv.r_plus(m_v, a)
    rs_v = kv.r_s(m_v)
    ax2.plot(rs_v*np.cos(theta), rs_v*np.sin(theta), '-', color=color, linewidth=1.8,
             label=f"v={v}M (m={m_v:.3f}M)")
    ax2.plot(rp_v*np.cos(theta), rp_v*np.sin(theta), '--', color=color, linewidth=1.2)

# fill only the LAST (most shrunk) horizon solid, for visual reference
m_last = max(m0 - alpha*v_snapshots[-1], 0.0)
rp_last = kv.r_plus(m_last, a)
ax2.fill(rp_last*np.cos(theta), rp_last*np.sin(theta), color='black', alpha=0.85)

ax2.set_title(f"Dynamic Kerr-Vaidya (a={a} fixed, alpha={alpha})")
ax2.set_xlim(-2.5, 2.5); ax2.set_ylim(-2.5, 2.5)
ax2.set_aspect('equal')
ax2.legend(loc='upper right', fontsize=7)

fig.tight_layout()
fig.savefig("fig_shrinking_ergosphere.pdf")
print("Saved fig_shrinking_ergosphere.pdf")

# print numbers for the caption
print("\nFor caption:")
for v in v_snapshots:
    m_v = max(m0 - alpha*v, 0.0)
    print(f"  v={v}M: m={m_v:.4f}M, r_+={kv.r_plus(m_v,a):.4f}M, r_s={kv.r_s(m_v):.4f}M, "
          f"a/m={a/m_v:.4f}")
plt.show()