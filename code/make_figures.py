"""Figures for the paper (vector PDF, print-oriented).
Palette: first three validated categorical slots (blue, orange, aqua), always
paired with distinct markers / line styles so identity never relies on colour.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kvcore as K

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})
FIG = "../figures/"
main = json.load(open("../data/study_all.json"))
via = json.load(open("../data/viability.json"))
fly = json.load(open("../data/flyby.json"))



def sel(law, spin):
    rows = [R for R in main if R["law"] == law and abs(R["spin"] - spin) < 1e-9]
    return sorted(rows, key=lambda R: R["alpha"])


# ------------------------------------------------------------- Fig: budget
def fig_budget(spin=0.9):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, law, name in zip(axs, (0, 1), ("law A (fixed $a$)", "law B (fixed $a/m$)")):
        rows = sel(law, spin)
        st = [R for R in sel(0, spin) if R["alpha"] == 0][0]["best"]
        dyn = [R for R in rows if R["alpha"] > 0 and R["best"] is not None]
        al = np.array([R["alpha"] for R in dyn])
        G = np.array([R["best"]["G"] for R in dyn])
        loc = np.array([R["best"]["loc"] for R in dyn])
        tin = np.array([R["best"]["T_in"] for R in dyn])
        tout = np.array([R["best"]["T_out"] for R in dyn])
        ax.axhline(st["G"], color=MUTED, ls=":", lw=1.0)
        ax.text(0.98, st["G"] + 0.003, "static Kerr optimum", color=MUTED, fontsize=7,
                va="bottom", ha="right", transform=ax.get_yaxis_transform())
        ax.plot(al, G, "-o", color=INK, label="net gain $G$")
        ax.plot(al, loc, "--s", color=C1, label=r"local $-E_2$")
        ax.plot(al, tin, "-.^", color=C2, label=r"transport in $\Delta E_{\rm in}$")
        ax.plot(al, tout, ":D", color=C3, label=r"transport out $\Delta E_{\rm out}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"mass-loss rate $\alpha=-\dot m$")
        ax.set_title(f"{name}, $a_0={spin}$", fontsize=9)
    axs[0].set_ylabel(r"energy per unit incoming mass")
    axs[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG + "fig_budget.pdf")
    plt.close(fig)


# ------------------------------------------------------------- Fig: locality
def fig_locality():
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    mk = {0: ("o", C1, "law A"), 1: ("s", C2, "law B")}
    for law in (0, 1):
        xs, ys = [], []
        for R in main:
            if R["law"] != law or R["best_loc"] is None:
                continue
            b = R["best_loc"]
            xs.append(b["bound"]); ys.append(b["eta_loc"])
        m_, c_, l_ = mk[law]
        ax.plot(xs, ys, m_, color=c_, mfc="none", mew=1.1, label=l_, ls="none")
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color=MUTED, lw=0.8, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"instantaneous bound $\eta_{\max}(E_1,r_s,m(v_s))$")
    ax.set_ylabel(r"best escaping decay $-E_2/E_1$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG + "fig_locality.pdf")
    plt.close(fig)


# ------------------------------------------------------------- Fig: flyby
def fig_flyby():
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    cases = [("lawA_a0.0_L4.75", C1, "o", "$a=0$ (Vaidya), $r_p=3.9M$"),
             ("lawA_a0.9_L3.9", C2, "s", "$a=0.9$, law A, $r_p=3.8M$"),
             ("lawA_a0.9_L3.0", C3, "^", "$a=0.9$, law A, $r_p=1.9M$")]
    for key, c, m_, lab in cases:
        d = fly[key]
        al = np.array([rw["alpha"] for rw in d["rows"]])
        dE = np.array([rw["dE"] for rw in d["rows"]])
        d1 = np.array([rw["dE_first_order"] for rw in d["rows"]])
        ax.plot(al, dE, m_, color=c, mfc="none", mew=1.0, label=lab, ls="none")
        ax.plot(al, d1, "-", color=c, lw=1.0)
        ax.axhline(d["dE_impulsive"], color=c, ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\Delta E=E_\infty-E_0$")
    ax.legend(loc="lower right", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(FIG + "fig_flyby.pdf")
    plt.close(fig)


# ------------------------------------------------------------- Fig: viability
def fig_viability():
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    a = np.array([r["a"] for r in via]); ac = np.array([r["alpha_crit"] for r in via])
    ap = np.array([r["alpha_pred"] for r in via])
    aa = np.linspace(0.78, 0.9995, 300)
    dvm = np.mean([r["dv_min"] for r in via])
    ax.fill_between(aa, 1e-3, (1 - aa) / dvm, color="#e3eefb", lw=0)
    ax.plot(aa, (1 - aa) / dvm, "-", color=C1, lw=1.2,
            label=rf"$(M-a)/\Delta v$, $\Delta v={dvm:.2f}M$")
    ax.plot(a, ap, "s", color=C1, mfc="none", mew=1.0, label=r"$(M-a)/\Delta v(a)$")
    ax.plot(a, ac, "o", color=C2, label=r"numerical $\alpha_{\rm crit}$, $E_0=1$")
    for fn, mk, lab in (("../data/viability_E0.95.json", "v", r"$E_0=0.95$"),
                        ("../data/viability_E1.2.json", "^", r"$E_0=1.2$")):
        d = json.load(open(fn))
        ax.plot([r["a"] for r in d], [r["alpha_crit"] for r in d], mk, color=C3, mfc="none",
                mew=1.0, label=r"numerical, " + lab)
        ax.plot([r["a"] for r in d], [r["alpha_pred"] for r in d], ":", color=C3, lw=0.9)
    ax.text(0.93, 4e-2, "viable", color=MUTED, fontsize=8)
    ax.text(0.93, 0.25, "no extraction", color=MUTED, fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.0); ax.set_xlim(0.78, 1.0)
    ax.set_xlabel(r"$a/M$"); ax.set_ylabel(r"$\alpha$")
    ax.legend(loc="lower left", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(FIG + "fig_viability.pdf")
    plt.close(fig)


# ------------------------------------------------------------- Fig: spin
def fig_spin():
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    aa = np.linspace(0.75, 0.9995, 300)
    rp = 1 + np.sqrt(1 - aa**2)
    ax.plot(aa, 0.5 * (np.sqrt(2 / rp) - 1), "-", color=MUTED, lw=0.9,
            label=r"BPT: $(\sqrt{2M/r_+}-1)/2$")
    for law, al, c, m_, lab in ((0, 0.0, INK, "o", r"static Kerr"),
                                (0, 1e-2, C1, "s", r"law A, $\alpha=10^{-2}$"),
                                (1, 1e-2, C2, "^", r"law B, $\alpha=10^{-2}$")):
        xs, ys = [], []
        for R in main:
            if R["law"] == law and R["alpha"] == al and R["best"] is not None:
                xs.append(R["best"]["a_s"] / R["best"]["m_s"]); ys.append(R["best"]["loc"])
        o = np.argsort(xs)
        ax.plot(np.array(xs)[o], np.array(ys)[o], "--" if law else "-", marker=m_, color=c,
                lw=1.0, label=lab)
    ax.set_xlabel(r"spin at the decay $a/m(v_s)$"); ax.set_ylabel(r"local gain $-E_{2s}$")
    ax.legend(loc="upper left", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(FIG + "fig_spin.pdf")
    plt.close(fig)


# ------------------------------------------------------------- Fig: example
def fig_example(law=0, spin=0.9, alpha=1e-3):
    R = [R for R in sel(law, spin) if abs(R["alpha"] - alpha) < 1e-12][0]
    b = R["best"]
    dm = R["dm"]
    Pst = np.array([float(law), 0.0, dm, spin, 1.0, 0.0])
    pr0 = K.pr_from_shell(10.0, 1.0, spin, -1.0, b["L0"], 1.0, "in")
    y0 = np.array([0.0, 10.0, 0.0, -1.0, pr0, b["L0"]])
    yon, _, _, _, tr0, n0 = K.integrate(y0, Pst, 1e5, 200, 2, 3.0, record=200000)
    P = np.array([float(law), alpha, dm, spin, 1.0, yon[0]])
    ys, _, st, _, tr1, n1 = K.integrate(yon, P, 1e5, 200, 1, b["delta"], record=200000)
    p2, p3 = K.decay(ys, P, 1.0, b["psi"])
    _, _, _, _, tr2, n2 = K.integrate(np.concatenate([ys[:3], p2]), P, 1e6, 200, 0, 0, record=400000)
    _, _, _, _, tr3, n3 = K.integrate(np.concatenate([ys[:3], p3]), P, 1e6, 60, 0, 0, record=400000)
    inc = np.vstack([tr0[:n0], tr1[:n1]])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9),
                            gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axs[0]
    th = np.linspace(0, 2 * np.pi, 300)
    rp0 = K.r_plus(1.0, spin); mf = 1 - dm
    rpf = K.r_plus(mf, spin if law == 0 else spin * mf)
    ax.fill(2 * np.cos(th), 2 * np.sin(th), color="#f3e3c3", lw=0)
    ax.fill(rp0 * np.cos(th), rp0 * np.sin(th), color="#3b3a37", lw=0)
    for tr, c, ls, lab in ((inc, C1, "-", "incoming"), (tr2[:n2], C2, "--", "fragment 2"),
                           (tr3[:n3], C3, "-.", "fragment 3")):
        ax.plot(tr[:, 1] * np.cos(tr[:, 2]), tr[:, 1] * np.sin(tr[:, 2]), ls, color=c,
                lw=1.2, label=lab)
    ax.plot(ys[1] * np.cos(ys[2]), ys[1] * np.sin(ys[2]), "x", color=INK, ms=6)
    ax.set_aspect("equal"); ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$r\cos\phi\;[M]$"); ax.set_ylabel(r"$r\sin\phi\;[M]$")
    ax.legend(loc="upper right", fontsize=6.5)
    ax = axs[1]
    von, vend = P[5], P[5] + dm / alpha
    ax.axvspan(von, vend, color="#ecebe7", lw=0)
    ax.text(von + 1.0, 0.55, "mass-loss\nepisode", fontsize=7, color=MUTED, va="center")
    ax.plot(inc[:, 0], -inc[:, 3], "-", color=C1, label="incoming")
    ax.plot(tr2[:n2, 0], -tr2[:n2, 3], "--", color=C2, label="fragment 2 (into hole)")
    ax.plot(tr3[:n3, 0], -tr3[:n3, 3], "-.", color=C3, label="fragment 3 (escapes)")
    ax.axvline(ys[0], color=MUTED, lw=0.6)
    ax.set_xlim(von - 8, max(vend, ys[0]) + 30)
    ax.set_xlabel(r"advanced time $v\;[M]$"); ax.set_ylabel(r"$E=-p_v$")
    ax.legend(loc="center right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG + "fig_example.pdf")
    plt.close(fig)
    return b


if __name__ == "__main__":
    fig_budget(0.9)
    fig_locality()
    fig_flyby()
    fig_viability()
    fig_spin()
    print(fig_example())
