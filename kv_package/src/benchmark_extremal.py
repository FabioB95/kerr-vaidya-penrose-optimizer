"""benchmark_extremal.py -- verifica il limite di Bardeen-Press-Teukolsky"""
import numpy as np
import sys
sys.path.insert(0, '.')
from split_extraction import max_extraction
import geodesics as kv
from scipy.optimize import minimize_scalar

# Replica la logica di calibration_test.py ma con E1=1 fissato esplicitamente
# e a/M che tende a 1, invece di orbita legata.
for a in [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:
    m = 1.0
    rp = kv.r_plus(m, a)
    eta_theory = 0.5*(np.sqrt(2*m/rp) - 1)
    margin = 1e-6
    r = rp + margin
    ext = max_extraction(r, m, a)
    if ext is None:
        print(f"a/M={a}: nessuno split trovato")
        continue
    E1 = 1.0  # fissato, non l'orbita legata del grid search
    eta_E1_fixed = ext['dE'] / E1
    print(f"a/M={a}: dE_max(geometria)={ext['dE']:.4f}  "
          f"eta(E1=1 fissato)={eta_E1_fixed*100:.4f}%  "
          f"eta_teorico(BPT)={eta_theory*100:.4f}%  "
          f"rapporto={eta_E1_fixed/eta_theory:.3f}")