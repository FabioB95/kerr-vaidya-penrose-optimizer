# Re-run the ORIGINAL (relaxed) pipeline exactly at its reported alpha_crit
# values and report the split time and mass: tests alpha_crit = (M-a)/v_split.
import sys, json
sys.path.insert(0, '/home/claude/fabiob95/kerr-vaidya-penrose-optimizer/kv_package/src')
import pipeline_v2 as p
rows = []
for a, ac in [(0.8, 0.070), (0.9, 0.036922832031250005), (0.95, 0.018759375), (0.99, 0.0038184667968750004)]:
    for fac in (0.98, 1.05):
        r = p.run_case_v2(a, ac * fac, drmin=0.001)
        if r is None:
            rows.append(dict(a=a, alpha=ac*fac, viable=False)); print(a, ac*fac, 'not viable'); continue
        row = dict(a=a, alpha=ac*fac, viable=True, v_split=r['v_split'], m_split=r['m_split'],
                   dE=r['dE'], v_ext=(1-a)/(ac*fac), pred=(1-a)/r['v_split'])
        rows.append(row); print(row)
json.dump(rows, open('/home/claude/kv_paper/data/old_crit_check.json', 'w'), indent=1)
