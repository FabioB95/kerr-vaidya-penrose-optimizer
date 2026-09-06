# Optimal Penrose Energy Extraction from Kerr–Vaidya Black Holes

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the numerical code and data used in the paper:

> **Optimal Energy Extraction from the Dynamic Ergosphere of Kerr–Vaidya Black Holes: A Phase-Diagram Approach**  
> F. Buffoli (2026)

We combine high-resolution geodesic integration with a two-stage variational optimizer to study the Penrose process in radiating Kerr–Vaidya spacetimes. The code finds optimal particle trajectories and computes the extraction efficiency η as a function of black-hole spin `a/M` and mass-loss rate `α`. We also determine the critical mass-loss rate `α_crit(a)` above which the Penrose process shuts down.

---

## Key Results

- **Static Kerr benchmark:** Recovers the theoretical bound η ≈ 20.7% for E₁ = 1; bound-orbit initial conditions yield higher formal efficiencies with modest absolute gains (≲ 0.8 M).
- **Dynamic Kerr–Vaidya:** Efficiency decreases monotonically with `α`.
- **Critical mass-loss rate:** For `a/M = 0.95`, `α_crit ≈ 7.9 × 10⁻⁴`.
- **Phase diagram:** First quantitative `(a, α)` boundary between viable and non-viable Penrose extraction.

---

## Repository Structure
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
│   ├── step01_kerr_geometry_spins.pdf
│   ├── step02_penrose_trajectory.pdf
│   ├── step04_optimizer_comparison.pdf
│   ├── step05_comprehensive_study.pdf
│   └── step06_phase_diagram.pdf
├── notebooks/
│   └── plot_figures.ipynb    # Jupyter notebook to reproduce all paper figures
├── tests/
│   └── test_convergence.py   # Convergence tests for the RK4 integrator
├── requirements.txt
└── README.md
plain

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/kerr-vaidya-penrose-optimizer.git
cd kerr-vaidya-penrose-optimizer
pip install -r requirements.txt
Requirements
Python ≥ 3.9
NumPy
SciPy
Matplotlib
SymPy
Jupyter (optional, for notebooks)
Usage
1. Run a single Penrose optimization
bash
python src/optimizer.py --spin 0.9 --alpha 0.0005 --output results.json
2. Reproduce the phase boundary
bash
python src/phase_boundary.py --spin 0.95 --alpha-max 0.01 --tolerance 2e-4
3. Generate all figures
bash
jupyter notebooks/plot_figures.ipynb
Citation
If you use this code, please cite:
bibtex
@article{buffoli2026penrose,
  author  = {Buffoli, Fabio},
  title   = {Optimal Energy Extraction from the Dynamic Ergosphere of {Kerr--Vaidya} Black Holes: A Phase-Diagram Approach},
 
}
License
This project is licensed under the MIT License. See LICENSE for details.
Contact
For questions or issues, please open a GitHub issue or contact:
Fabio Buffoli
Università degli Studi di Brescia, Italy
fabio.buffoli@unibs.it
plain

---

## GitHub Project Description (for the "About" section)

> Numerical study of optimal Penrose energy extraction in Kerr–Vaidya spacetimes. High-resolution geodesic integration + variational optimizer to map the (a, α) phase diagram and locate the critical mass-loss rate beyond which extraction fails.

---

**Tip:** Add a `LICENSE` file (MIT is standard for academic code) and a `requirements.txt` before you link the repo in the paper. Good luck with the submission — this is a solid piece of work.
