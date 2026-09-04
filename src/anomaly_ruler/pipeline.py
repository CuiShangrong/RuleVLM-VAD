from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .core import (OFFICIAL_ANOMALY_KEYWORDS, append_jsonl, atomic_json, cluster_keyword,
                   official_final_scores, official_perception_smoothing, pad_last,
                   parse_final_decision, sha256_file, stable_hash)
from .data import SETUPS, ShanghaiData
from .prompts import correction_messages, deduction_prompt, induction_messages
from .qwen_backend import QwenBackend


def _parallel_ordered(fn, items, workers, progress=None):
    out=[None]*len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,item):i for i,item in enumerate(items)}
        done=0
        for future in as_completed(futures):
            i=futures[future]; out[i]=future.result(); done+=1
            if progress and (done % 25 == 0 or done == len(items)): progress(done,len(items))
    return out


def build_rule(workspace, run_root, source, backend, workers=8):
    data=ShanghaiData(workspace); out=Path(run_root)/'knowledge'/f'rule_{source}'
    out.mkdir(parents=True,exist_ok=True)
    rule_path=out/'rule.txt'
    if rule_path.exists() and (out/'rule_manifest.json').exists(): return rule_path
    ids=data.source_normal_ids(source)
    if not ids: raise ValueError(f'No normal source videos for {source}')
    all_frames=[]
    for vid in ids: all_frames.extend((vid,p) for p in data.source_frames(vid))
    rng=np.random.RandomState(2024)
    chosen=rng.choice(len(all_frames),10,replace=False).tolist()
    samples=[all_frames[i] for i in chosen]
    results=_parallel_ordered(lambda item: backend.describe_image(item[1]),samples,workers)
    if not all(x.get('success') for x in results): raise RuntimeError('source visual perception failed')
    descriptions=[x['text'] for x in results]
    induction=backend.chat(induction_messages(descriptions),'induction',{'source':source,'descriptions':descriptions})
    if not induction.get('success'): raise RuntimeError(induction)
    correction=backend.chat(correction_messages([induction['text']],1),'rule_correction',
                            {'source':source,'rules':[induction['text']]})
    if not correction.get('success'): raise RuntimeError(correction)
    rule_path.write_text(correction['text'].strip()+'\n')
    manifest={"source":source,"seed":2024,"official_b":1,"official_bs":10,
      "normal_source_video_ids":ids,"selected_frames":[{"video_id":v,"path":str(p)} for v,p in samples],
      "descriptions":descriptions,"induction_response":induction,"correction_response":correction,
      "rule_sha256":sha256_file(rule_path)}
    atomic_json(out/'rule_manifest.json',manifest)
    (out/'source_normal_video_ids.txt').write_text('\n'.join(ids)+'\n')
    return rule_path


def _describe_video(backend, frames, workers, log):
    start=time.monotonic()
    def progress(n,total):
        append_jsonl(log,{"stage":"description","completed":n,"total":total,"elapsed":time.monotonic()-start})
    results=_parallel_ordered(backend.describe_image,frames,workers,progress)
    failed=[(i,x) for i,x in enumerate(results) if not x.get('success')]
    if failed: raise RuntimeError(f'{len(failed)} image descriptions failed; first={failed[0]}')
    return [x['text'] for x in results],results


def run_inference(workspace, run_root, setup, backend, *, workers=8, video_ids=None, smoke=False, attempt=None):
    if setup not in SETUPS: raise ValueError(setup)
    source,target=SETUPS[setup]; data=ShanghaiData(workspace)
    leaf=setup if not attempt else f'{setup}_{attempt}'
    out=Path(run_root)/('smoke' if smoke else 'setups')/leaf
    out.mkdir(parents=True,exist_ok=True); (out/'logs').mkdir(exist_ok=True)
    # A locked prediction is immutable.  Resume never silently regenerates it.
    if (out/'prediction_lock.json').exists() and (out/'frame_scores.npy').exists():
        return out
    rule_path=Path(run_root)/'knowledge'/f'rule_{source}'/'rule.txt'
    if not rule_path.exists(): raise FileNotFoundError(rule_path)
    rule=rule_path.read_text()
    manifest=data.target_manifest(target)
    if video_ids is not None:
        wanted=set(video_ids); manifest=[x for x in manifest if x['video_id'] in wanted]
        if len(manifest)!=len(wanted): raise ValueError('smoke video not in target split')
    sanitized=[{"video_id":x['video_id'],"frame_count":x['frame_count'],
                "prediction_length":x['gt_end']-x['gt_start']} for x in manifest]
    atomic_json(out/'inference_config.json',{"algorithm":"AnomalyRuler official core with Qwen3.8 backend",
      "official_commit":"210ce0b5470152ac0361691706bf49e291e1c47d","setup":setup,"source":source,
      "target":target,"seed":2024,"endpoint":backend.endpoint,"model":backend.model,
      "revision":backend.revision,"rule_path":str(rule_path),"rule_sha256":sha256_file(rule_path),
      "workers":workers,"smoke":smoke,"target_manifest":sanitized,"gt_read":False})
    (out/'target_test_video_ids.txt').write_text('\n'.join(x['video_id'] for x in sanitized)+'\n')
    all_scores=[]; video_records=[]
    raw_path=out/'raw_qwen_responses.jsonl'; desc_path=out/'frame_descriptions.jsonl'
    for vi,item in enumerate(manifest):
        vid=item['video_id']; frames=data.test_frames(vid); log=out/'logs'/f'{vid}.jsonl'
        descriptions,desc_results=_describe_video(backend,frames,workers,log)
        for i,(text,res) in enumerate(zip(descriptions,desc_results)):
            append_jsonl(desc_path,{"video_id":vid,"frame_index":i,"description":text,
                                   "cache_hit":res.get('cache_hit'),"elapsed_seconds":res.get('elapsed_seconds')})
        smooth=official_perception_smoothing(descriptions)
        modified=smooth['modified_descriptions']; initial=smooth['raw_keyword_predictions']
        jobs=[]
        for i,(description,pred) in enumerate(zip(modified,initial)):
            hits=[k for k in OFFICIAL_ANOMALY_KEYWORDS if k in description]
            prompt=deduction_prompt(description,rule,int(pred),hits[0] if hits else None)
            messages=[{"role":"user","content":prompt}]
            jobs.append((i,int(pred),messages,{"setup":setup,"rule_sha256":sha256_file(rule_path),
                         "video_id":vid,"frame_index":i,"description":description,"initial_pred":int(pred)}))
        def deduct(job):
            i,ini,msg,key=job; response=backend.chat(msg,'deduction',key,max_tokens=4000)
            if not response.get('success'): return {"frame_index":i,"initial_pred":ini,"response":response,"success":False}
            try: parsed=parse_final_decision(response['text'])
            except Exception as exc: return {"frame_index":i,"initial_pred":ini,"response":response,"success":False,"parse_error":repr(exc)}
            # Exact official double-deduction override: positive keyword initial result remains positive.
            return {"frame_index":i,"initial_pred":ini,"llm_pred":parsed,"prediction":1 if ini else parsed,
                    "response":response,"success":True}
        start=time.monotonic()
        def progress(n,total): append_jsonl(log,{"stage":"deduction","completed":n,"total":total,"elapsed":time.monotonic()-start})
        deductions=_parallel_ordered(deduct,jobs,workers,progress)
        failures=[x for x in deductions if not x['success']]
        if failures: raise RuntimeError(f'{vid}: {len(failures)} deduction failures; first={failures[0]}')
        for x in deductions: append_jsonl(raw_path,{"video_id":vid,**x})
        preds=np.asarray([x['prediction'] for x in deductions],dtype=np.int8)
        raw_scores=official_final_scores(preds)
        expected=item['gt_end']-item['gt_start']; scores=pad_last(raw_scores,expected).astype(np.float32)
        np.save(out/f'{vid}_frame_scores.npy',scores)
        all_scores.extend(scores.tolist())
        record={"video_id":vid,"raw_frame_count":len(frames),"prediction_length":len(scores),
          "padding":len(scores)-len(frames),"keyword_threshold":smooth['threshold'],
          "raw_keyword_positive":int(initial.sum()),"final_positive":int(preds.sum()),
          "description_cache_hits":sum(bool(x.get('cache_hit')) for x in desc_results),
          "deduction_cache_hits":sum(bool(x['response'].get('cache_hit')) for x in deductions)}
        video_records.append(record); append_jsonl(out/'video_results.jsonl',record)
    arr=np.asarray(all_scores,dtype=np.float32)
    if not np.isfinite(arr).all() or (arr<0).any() or (arr>1).any(): raise ValueError('invalid predictions')
    np.save(out/'frame_scores.npy',arr)
    atomic_json(out/'inference_manifest.json',{"setup":setup,"videos":video_records,"prediction_length":len(arr),
      "frame_scores_sha256":sha256_file(out/'frame_scores.npy'),"gt_read":False,"completed_at":time.time()})
    lock={"setup":setup,"prediction":str(out/'frame_scores.npy'),"sha256":sha256_file(out/'frame_scores.npy'),
          "length":len(arr),"rule_sha256":sha256_file(rule_path),"gt_read_before_lock":False,"locked_at":time.time()}
    atomic_json(out/'prediction_lock.json',lock)
    return out
