#!/usr/bin/env python3
import argparse,sys,json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'UR-DMU'))
from anomaly_ruler.evaluate import evaluate_locked
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();print(json.dumps(evaluate_locked(a.workspace,a.output_dir),indent=2))
