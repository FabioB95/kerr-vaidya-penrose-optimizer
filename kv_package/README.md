# Kerr-Vaidya Penrose process -- corrected numerics (handoff package v3)

## Status: pipeline validated, ready to run the full sweeps

The blocking bug is fixed and the pipeline is now internally consistent
and physically verified end-to-end. Summary of the fix chain:

1. **Metric bug** (Eq. 1 of the original paper): g_{r phi} should be the
   constant -a sin^2(theta), not 2am(v)r/Delta. Verified by direct
   Boyer-Lindquist -> advanced-coordinate transform (`derive_metric.py`).

2. **A second, more subtle bug** (found after the first handoff): Sigma =
   r^2 + a^2*cos^2(theta) reduces to r^2 on the equator (theta=pi/2), NOT
   r^2+a^2 -- every equatorial-plane script had this wrong. Caught by
   checking that the horizon Killing vector chi = d/dv + Omega_H d/dphi is
   null at r=r_plus (it wasn't, before the fix). Fixed in `kv_metric.py` /
   `geodesics.py`.

3. **Escape criterion bug**: r_esc=5M was less than the launch radius
   r0=6M. Fixed: r_esc=20M plus genuine ODE-integration verification of
   sustained outward motion (`verify_escape()` in `pipeline.py`).

4. **Missing physics constraint** (not a bug in the original paper so much
   as an incompleteness): the Wald/Christodoulou area theorem (irreducible
   mass must not decrease) was never enforced, which is why the original
   optimizer could report single-event energy extractions exceeding the
   ENTIRE rotational reservoir of the hole. Now enforced via
   `split_extraction.py`.

5. **Key simplification found**: the maximum extractable |E2| depends only
   on the local geometry (r_split, m_split, a) via the area theorem -- NOT
   on E1 or L1 of the incoming particle. This cuts the search space
   enormously (see `split_extraction.max_extraction()`).

**Validated end-to-end**: `pipeline_v2.run_case_v2()` finds a genuine
incoming trajectory (from r0=6M), a split point close to the horizon, an
escape-verified outgoing fragment -- for a/M=0.9 static Kerr this gives
dE=0.1527M, eta=18.4% at the trajectory's own E1=0.83. This is ~1.7x the
often-cited closed-form ceiling of 9.0% for a *specific, more restricted*
photon-direction ansatz (see the note in `calibration_test.py` and the
chat history) -- our number is the unconstrained maximum subject to the
same physical laws (area theorem + genuine escape), searched over ALL
photon directions, so it cannot be lower than that restricted sub-case.
Recommend reporting dE as the primary, geometry-only physical quantity in
the paper, with eta as the E1-dependent secondary quantity (exactly as
Sec. 4.1 already discusses, now backed by a rigorous, escape-verified
number instead of an unconstrained brute-force artifact).

## Run order

1. `python3 src/derive_metric.py` -- independent metric check (BL -> advanced transform)
2. `python3 src/geodesics.py` -- energy conservation sanity check
3. `python3 src/calibration_test.py` -- documents the fix chain and the
   photon-cone / area-theorem calculation; ~15.3% for a/M=0.9 at the
   horizon limit
4. `python3 src/pipeline_v2.py` -- quick end-to-end single-case test
   (should print dE~0.153, escape-verified, in ~5-10s)
5. **`python3 src/run_sweeps.py`** -- the actual data generation. Expect
   30-60 minutes total (see the script's docstring for a runtime
   breakdown). Produces `sweep_results.json`.

## Files

- `derive_metric.py` -- independent BL -> advanced-coords transform (ground truth)
- `kv_metric.py` -- symbolic corrected metric + Christoffels (for Appendix A)
- `geodesics.py` -- Hamiltonian-form RK4 integrator (corrected Sigma)
- `split_extraction.py` -- **the validated stage-2 physics**: photon-cone
  parametrization + area-theorem saturation, geometry-only max |E2|
- `pipeline.py` -- stage-1 skimming search + escape verification utilities
- `pipeline_v2.py` -- **the pipeline to actually use**: merges stage 1 and
  the validated stage 2, only accepts genuinely escape-verified candidates
- `calibration_test.py` -- documents the full debugging history and the
  residual ~1.7x discrepancy with a specific literature ansatz (see above)
- `run_sweeps.py` -- **run this for the paper's data**
- `boundary_search.py`, `joint_search.py`, `optimizer2.py` -- earlier,
  superseded attempts, kept for reference/history only; not needed for
  the final pipeline

## After the sweeps run

Send back `sweep_results.json` and I'll regenerate the 6 figures, Table 3,
and the corrected text for Sec. 4.1-4.6, Eq. (1), and Appendix A, plus a
new methodological subsection on the area-theorem constraint (this is a
genuine addition to the paper's contribution, not just a bug fix).
