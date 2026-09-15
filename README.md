# Penrose Process in Kerr–Vaidya Spacetimes

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the numerical code and data used in the paper:

> **An Operative Viability Boundary for Relaxed Single-Particle Penrose Extraction in Kerr–Vaidya Spacetimes**  
> F. Buffoli (2026), MNRAS (submitted)

We combine high-resolution geodesic integration with a two-stage variational optimizer to study the Penrose process in radiating Kerr–Vaidya spacetimes. The code finds optimal particle trajectories, computes the extraction efficiency η and the physically bounded energy gain ΔE as functions of black-hole spin `a/M` and mass-loss rate `α`, and determines the operative viability boundary `α_crit(a)` above which no escape-verified split is found.

---

## Key Results

- **Static Kerr benchmark:** ΔE increases monotonically with spin, from ΔE ≈ 0.106 M at a/M = 0.80 to ΔE ≈ 0.245 M at a/M = 0.99. The pipeline reproduces the classical Bardeen–Press–Teukolsky bound η_BPT ≈ 20.7% to within 0.1% in the idealized horizon limit (full 4-momentum conservation at the split).
- **Effective-spin drift (central result):** Because the fixed-`a` Kerr–Vaidya construction holds `a` fixed while `m(v)` decreases, the effective dimensionless spin `a/m(v)` drifts substantially toward extremality as the hole radiates — from `a/m_split = 0.81` at nominal `a/M = 0.80` to `a/m_split = 0.9999` at nominal `a/M = 0.99`. This is a purely geometric feature of the construction and is independent of the split formulation.
- **Operative viability boundary `α_crit(a)`:** Using the physical mass-function limit `m ≥ 0` (complete evaporation), we find `α_crit(0.80) ≈ 0.070`, `α_crit(0.90) ≈ 0.036`, `α_crit(0.95) ≈ 0.019`, `α_crit(0.99) ≈ 0.0038` — a trend that **decreases** with spin. The boundary is insensitive to the horizon safety margin `Δr_min` once the physical mass floor is enforced.
- **Proper-time budget hypothesis falsified:** The decreasing trend of `α_crit` with spin is *not* explained by a naive proper-time budget effect — the product `α_crit · τ_cross` decreases with spin while the prediction `(r_s − r_+)/2` increases. This is reported as an open question in the paper.
- **Hawking evaporation never reaches the boundary:** For any physically realizable black hole, the Hawking mass-loss rate `α ~ ℏ/m²` is 40–70 orders of magnitude below `α_crit`; the boundary is relevant only to non-Hawking mass-loss scenarios.

---

## Repository Structure

```
.
├── src/
│   ├── geodesics.py          # RK4 geodesic integrator for equatorial Kerr–Vaidya
│   ├── optimizer.py          # Two-stage variational optimizer (skimming + split)
│   ├── phase_boundary.py     # Binary search for α_crit(a)
│   └── christoffel.py        # Symbolic Christoffel symbols (SymPy)
├── data/
│   ├── alpha_sweep/          # Efficiency vs. α data files
│   ├── spin_sweep/           # Efficiency vs. a/M data files
│   └── phase_boundary/       # Binary-search convergence logs
├── figures/
│   ├── fig_kerr_geometry.pdf
│   ├── fig_trajectories_static_vs_dynamic.pdf
│   ├── fig_shrinking_ergosphere.pdf
│   ├── fig_alpha_sweep.pdf
│   ├── fig_spin_dependence.pdf
│   └── fig_phase_boundary.pdf
├── notebooks/
│   └── plot_figures.ipynb    # Jupyter notebook to reproduce all paper figures
├── tests/
│   └── test_convergence.py   # Convergence tests for the RK4 integrator
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/FabioB95/kerr-vaidya-penrose-optimizer.git
cd kerr-vaidya-penrose-optimizer
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9
- NumPy
- SciPy
- Matplotlib
- SymPy
- Jupyter (optional, for notebooks)

---

## Usage

### 1. Run a single Penrose optimization

```bash
python src/optimizer.py --spin 0.9 --alpha 0.0005 --output results.json
```

### 2. Reproduce the phase boundary

```bash
python src/phase_boundary.py --spin 0.90 --alpha-max 0.10 --tolerance 2e-4
```

### 3. Generate all figures

```bash
jupyter notebook notebooks/plot_figures.ipynb
```

---

## Method Summary

The optimizer proceeds in two stages:

1. **Stage 1 — Skimming orbit search.** A grid search over initial parameters `(E, L, p_r,0)` identifies trajectories that penetrate the ergosphere deeply while maintaining a minimum safe distance `Δr_min = r − r_+ ≥ 0.3 M` from the horizon. The score function rewards both depth of penetration and dwell time inside the ergosphere.

2. **Stage 2 — Brute-force split optimization.** At the deepest safe point inside the ergosphere, we search over split parameters `(E₃, L₃, p_r,3)` to maximize the extraction efficiency η, subject to:
   - energy and angular momentum conservation (`E₁ = E₂ + E₃`, `L₁ = L₂ + L₃`),
   - the effective-potential condition `V_eff(r_split) ≤ 0` for both fragments,
   - the Wald/Christodoulou area theorem (irreducible mass of the resulting black hole must not decrease),
   - verified escape of fragment 3 to `r > 20 M` via forward integration with high resolution.

**Important caveat on the split formulation.** All quantitative results in the paper use the standard single-particle treatment, in which the split conserves `E` and `L` but *not* the escaping fragment's radial momentum `p_r`. This is the standard practice in the single-particle Penrose process literature. The paper also reports a separate validation against the Bardeen–Press–Teukolsky bound using full 4-momentum conservation at the split, in the idealized horizon limit; see Section 4.1 of the paper.

---

## Data Availability

The data files underlying the figures of the paper are provided in the `data/` directory:

- `data/alpha_sweep/` — energy gain ΔE and efficiency η vs. `α` for `a/M = 0.9`
- `data/spin_sweep/` — ΔE and η vs. `a/M` for static and dynamic cases
- `data/phase_boundary/` — binary-search convergence logs for `α_crit(a)` at each spin

All results were generated with the code in `src/`, using the numerical parameters reported in Table 2 of the paper.

---

## Citation

If you use this code, please cite:

```bibtex
@article{buffoli2026penrose,
  author  = {Buffoli, Fabio},
  title   = {An Operative Viability Boundary for Relaxed Single-Particle
             {Penrose} Extraction in {Kerr--Vaidya} Spacetimes},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year    = {2026},
  note    = {submitted}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact:

**Fabio Buffoli**  
Università degli Studi di Brescia, Italy  
fabio.buffoli@unibs.it
