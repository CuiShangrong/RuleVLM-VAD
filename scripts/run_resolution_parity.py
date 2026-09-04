#!/usr/bin/env python3
"""Compare the same fixed old-vLLM clips with native-resolution Transformers."""
import json,sys
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'vendor'),str(Path(__file__).resolve().parents[3]/'UR-DMU')]
from anomaly_ruler.data import ShanghaiData
from anomaly_ruler.core import atomic_json
from vlm_baselines.fair_unified import UnifiedScorer,FAIR

p=Path(__file__).resolve().parents[1]; w=Path('/workspace'); old=w/'direct_vlm/zero shot/runs/dense_tokenprob_10s/targets'; out=w/'direct_vlm/fair_tuned/fair_runs/resolution_parity_native';out.mkdir(parents=True,exist_ok=True)
model=w/'vlm_service/qwen3_8_27b/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
cfg={**FAIR,'min_pixels':65536,'max_pixels':16777216}; d=ShanghaiData(w);s=UnifiedScorer(str(model),cfg=cfg)
rows=[]
for target in ['domain_01','domain_08']:
    x=[json.loads(z) for z in (old/target/'center_manifest.jsonl').read_text().splitlines()]
    for j in np.linspace(0,len(x)-1,10,dtype=int):
        r=x[int(j)]; paths=d.test_frames(r['video_id']); new=s.score([str(paths[i]) for i in r['sample_indices']],r['sample_indices'])
        rows.append({'target':target,'video_id':r['video_id'],'center_frame':r['center_frame'],'sample_indices':r['sample_indices'],'vllm_score':r['score'],'vllm_logprob_0':r['logprob_0'],'vllm_logprob_1':r['logprob_1'],'transformers_score':new['score'],'transformers_logit':new['logit']})
        (out/'clips.jsonl').open('a').write(json.dumps(rows[-1])+'\n')
for target in ['domain_01','domain_08','all']:
    z=[r for r in rows if target=='all' or r['target']==target];a=np.array([r['vllm_score'] for r in z]);b=np.array([r['transformers_score'] for r in z]);
    for i,r in enumerate(z):r['vllm_rank']=int(np.argsort(np.argsort(a))[i]);r['transformers_rank']=int(np.argsort(np.argsort(b))[i])
    atomic_json(out/f'{target}_summary.json',{'clips':len(z),'pearson':float(np.corrcoef(a,b)[0,1]),'spearman_rank':float(np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))[0,1]),'mae':float(np.abs(a-b).mean()),'native_processor':cfg})
atomic_json(out/'parity_complete.json',{'clips':20,'native_processor':cfg})
