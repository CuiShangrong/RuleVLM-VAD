"""Dense, target-only Direct Qwen VAD using forced-choice token probabilities."""
from __future__ import annotations
import base64, hashlib, json, math, time, urllib.request
from pathlib import Path
import numpy as np

from anomaly_ruler.core import atomic_json, append_jsonl, pad_last, sha256_file, stable_hash
from anomaly_ruler.data import ShanghaiData
from .common import ZERO_SHOT_TARGETS, uniform_indices_fixed_count

SYSTEM = 'You are a visual anomaly detector. Output only 0 or 1.'
PROMPT = ('You are given eight chronological surveillance frames from one approximately 10-second clip. '
          'Based only on visible evidence, classify the clip. Focus on human actions, interactions, vehicles, '
          'falls, fights, abnormal motion, and abnormal objects. Do not infer unseen events. '
          'Output exactly one character: 0 for normal or 1 for abnormal.')

def temporal_centers(length: int, stride: int = 16):
    if length <= 0: raise ValueError('empty video')
    out=list(range(0, length, stride))
    if out[-1] != length-1: out.append(length-1)
    return out

def context_bounds(center: int, length: int, context: int = 240):
    left=context//2; right=context-left
    return max(0, center-left), min(length, center+right)

def interpolate_centers(centers, values, length):
    if len(centers) != len(values) or not centers: raise ValueError('invalid centers')
    return np.interp(np.arange(length), np.asarray(centers), np.asarray(values)).astype(np.float32)

class TokenProbabilityScorer:
    def __init__(self, endpoint, model, revision, cache_root, timeout=300):
        self.endpoint, self.model, self.revision = endpoint, model, revision
        self.cache_root=Path(cache_root); self.cache_root.mkdir(parents=True,exist_ok=True); self.timeout=timeout
    @staticmethod
    def _url(p): return 'data:image/jpeg;base64,'+base64.b64encode(Path(p).read_bytes()).decode()
    def score(self, paths, indices):
        key=stable_hash({'model':self.model,'revision':self.revision,'system':SYSTEM,'prompt':PROMPT,
          'frames':[{'i':i,'sha256':sha256_file(p)} for i,p in zip(indices,paths)],'scoring':'token_logprob_0_vs_1'})
        file=self.cache_root/'dense_tokenprob'/f'{key}.json'
        if file.exists():
            x=json.loads(file.read_text())
            if x.get('success'): x['cache_hit']=True; return x
        content=[{'type':'text','text':PROMPT}]
        for i,p in zip(indices,paths): content += [{'type':'text','text':f'Original frame index: {i}'},{'type':'image_url','image_url':{'url':self._url(p)}}]
        payload={'model':self.model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':content}],
          'temperature':0,'max_tokens':1,'logprobs':True,'top_logprobs':20,'chat_template_kwargs':{'enable_thinking':False}}
        failures=[]; start=time.monotonic()
        for attempt in range(2):
            try:
                req=urllib.request.Request(self.endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer EMPTY'})
                with urllib.request.urlopen(req,timeout=self.timeout) as r: body=json.loads(r.read())
                ch=body['choices'][0]; item=ch['logprobs']['content'][0]
                table={z['token']:float(z['logprob']) for z in item['top_logprobs']}
                if '0' not in table or '1' not in table: raise ValueError(f'missing 0/1 logprob: {list(table)}')
                l0,l1=table['0'],table['1']; score=1/(1+math.exp(l0-l1))
                out={'success':True,'score':score,'forced_choice':ch['message']['content'].strip(),'logprob_0':l0,'logprob_1':l1,
                  'raw_choice':ch,'elapsed_seconds':time.monotonic()-start,'cache_hit':False,'key':key,'attempt':attempt+1}
                break
            except Exception as e: failures.append({'attempt':attempt+1,'error':repr(e)})
        else: out={'success':False,'attempts':failures,'error':failures[-1]['error'],'elapsed_seconds':time.monotonic()-start,'cache_hit':False,'key':key}
        atomic_json(file,out); return out

def run_dense_target(workspace, run_root, target_name, scorer, *, context_frames=240, center_stride=16, video_ids=None, smoke=False, extra_config=None):
    if target_name not in ZERO_SHOT_TARGETS: raise ValueError(target_name)
    data=ShanghaiData(workspace); target=ZERO_SHOT_TARGETS[target_name]
    out=Path(run_root)/('smoke' if smoke else 'targets')/target_name; out.mkdir(parents=True,exist_ok=True)
    if (out/'prediction_lock.json').exists() and (out/'frame_scores.npy').exists(): return out
    manifest=data.target_manifest(target)
    if video_ids:
        want=set(video_ids); manifest=[x for x in manifest if x['video_id'] in want]
        if len(manifest)!=len(want): raise ValueError('requested video absent')
    clean=[{'video_id':x['video_id'],'frame_count':x['frame_count'],'prediction_length':x['gt_end']-x['gt_start']} for x in manifest]
    cfg={'method':'Direct Qwen3.8-27B dense forced-choice token-probability zero-shot','target_name':target_name,'target_domain':target,
      'source_data_used':False,'source_domain':None,'model':scorer.model,'revision':scorer.revision,'endpoint':scorer.endpoint,
      'system_prompt':SYSTEM,'prompt':PROMPT,'prompt_sha256':hashlib.sha256((SYSTEM+PROMPT).encode()).hexdigest(),
      'sampling':{'context_frames':context_frames,'nominal_fps':24,'nominal_context_seconds':context_frames/24,'center_stride_frames':center_stride,'sample_count':8,'boundary_policy':'truncate_valid_context_and_repeat_only_image_slots'},
      'scoring':{'forced_choices':['0','1'],'continuous_score':'exp(logP1)/(exp(logP0)+exp(logP1))','temperature':0,'max_new_tokens':1,'top_logprobs':20},
      'recovery':{'method':'linear_interpolation_at_temporal_centers','extra_smoothing':'none'},'seed':2022,'target_manifest':clean,'gt_read':False,'smoke':smoke,**(extra_config or {})}
    atomic_json(out/'config.json',cfg);atomic_json(out/'inference_config.json',cfg);(out/'target_test_video_ids.txt').write_text('\n'.join(x['video_id'] for x in clean)+'\n')
    all_scores=[]; videos=[]; total_vlm_calls=0; total_vlm_elapsed=0.0; run_started=time.monotonic()
    for it in manifest:
        vid=it['video_id']; paths=data.test_frames(vid); expected=it['gt_end']-it['gt_start']; pv=out/f'{vid}_frame_scores.npy'
        if pv.exists():
            a=np.load(pv)
            if a.shape==(expected,) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(): all_scores.extend(a.tolist());videos.append({'video_id':vid,'raw_frame_count':len(paths),'prediction_length':len(a),'resumed':True});continue
        cs=temporal_centers(len(paths),center_stride); vals=[]
        for c in cs:
            lo,hi=context_bounds(c,len(paths),context_frames); ids=uniform_indices_fixed_count(lo,hi,8); r=scorer.score([paths[i] for i in ids],ids)
            if not r['success']: raise RuntimeError(f'{vid} center {c}: {r}')
            total_vlm_calls += 1; total_vlm_elapsed += float(r.get('elapsed_seconds', 0.0))
            rec={'video_id':vid,'center_frame':c,'context_start':lo,'context_end_exclusive':hi,'sample_indices':ids,'sample_count':8,**r};append_jsonl(out/'center_manifest.jsonl',rec);append_jsonl(out/'raw_vlm_responses.jsonl',rec);vals.append(r['score'])
        raw=interpolate_centers(cs,vals,len(paths)); a=pad_last(raw,expected).astype(np.float32);np.save(pv,a);all_scores.extend(a.tolist());videos.append({'video_id':vid,'raw_frame_count':len(paths),'prediction_length':len(a),'centers':len(cs),'resumed':False})
    scores=np.asarray(all_scores,dtype=np.float32)
    if not np.isfinite(scores).all() or not ((scores>=0)&(scores<=1)).all(): raise ValueError('invalid score')
    np.save(out/'frame_scores.npy',scores);atomic_json(out/'inference_manifest.json',{'videos':videos,'prediction_length':len(scores),'frame_scores_sha256':sha256_file(out/'frame_scores.npy'),'gt_read':False,'target_order':[x['video_id'] for x in clean], 'vlm_calls':total_vlm_calls, 'sum_vlm_elapsed_seconds':total_vlm_elapsed, 'wall_elapsed_seconds':time.monotonic()-run_started})
    atomic_json(out/'prediction_lock.json',{'target_name':target_name,'prediction':str(out/'frame_scores.npy'),'sha256':sha256_file(out/'frame_scores.npy'),'length':len(scores),'gt_read_before_lock':False})
    return out
