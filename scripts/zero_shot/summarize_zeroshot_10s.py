#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--run-root',required=True); a=p.parse_args()
root=Path(a.run_root); rows=[]
# Existing locked UR-DMU same-target controls, reported with source paths when found.
baseline={'domain_01':0.772954,'domain_08':0.817268,'mixed':0.929528}
for target in ('domain_01','domain_08','mixed'):
    path=root/'targets'/target/'evaluation.json'
    if not path.exists(): raise FileNotFoundError(path)
    m=json.loads(path.read_text())
    rows.append({'setup':f'Zero-shot -> {target.replace("domain_", "Domain ").replace("mixed", "Full ShanghaiTech / Mixed")}',
                 'target':target,'frame_auroc':m['frame_auroc'],'frame_ap':m['frame_ap'],
                 'macro_auroc':m['macro_auroc'],'videos':m['videos'],
                 'prediction_frames':m['prediction_frames'],'elapsed_note':'see inference manifest',
                 'urdmu_same_target_auroc':baseline[target],
                 'auroc_delta_vs_urdmu':m['frame_auroc']-baseline[target]})
with (root/'results_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/'results_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
