"""add_alpha_crit_099.py -- run once in kv_package/src/"""
import json
import numpy as np
import sys
sys.path.insert(0, '.')
from pipeline_v2 import run_case_v2

with open("sweep_results.json") as f:
    OUT = json.load(f)

def alpha_crit_binary_search(a, alpha_min=1e-5, alpha_max=0.3, tol=2e-4, max_iter=16):
    lo, hi = alpha_min, alpha_max
    tries = 0
    while run_case_v2(a=a, alpha=hi, drmin=0.001) is not None and tries < 6:
        print(f"  still viable at alpha={hi}, widening...")
        hi *= 2
        tries += 1
    if run_case_v2(a=a, alpha=lo, drmin=0.001) is None:
        print(f"  WARNING: not viable even at alpha_min={lo}")
        return None
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if run_case_v2(a=a, alpha=mid, drmin=0.001) is not None:
            lo = mid
        else:
            hi = mid
    return lo

crit = alpha_crit_binary_search(0.99)
print("alpha_crit(0.99) =", crit)
OUT["alpha_crit_a0.99"] = crit

with open("sweep_results.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("Updated sweep_results.json")