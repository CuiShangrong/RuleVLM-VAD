from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from anomaly_ruler.core import atomic_json, append_jsonl, pad_last, sha256_file, stable_hash
from anomaly_ruler.data import SETUPS, ShanghaiData

DIRECT_SYSTEM = "You are a surveillance-video anomaly scorer. Use only visible evidence in the chronological images. Do not assume a video label or unseen event. Output JSON only; no markdown."
DIRECT_PROMPT = """These are chronological RGB frames from a campus surveillance window. Rate visible anomaly evidence continuously.
Return exactly {"anomaly_score": integer 0-100, "confidence": number 0-1, "decision": "normal"|"uncertain"|"abnormal", "visible_evidence": [strings]}."""


def uniform_indices(start, end, count):
    if end <= start: raise ValueError('empty window')
    return [int(x) for x in np.linspace(start,end-1,min(count,end-start)).round()]


def uniform_indices_fixed_count(start, end, count):
    """Return exactly ``count`` chronological indices, repeating for short tails.

    A tail window is still an observation of one clip.  It must therefore carry
    eight image slots just like a full window; duplicated indices are explicit
    padding and are never used to extend the reconstructed frame sequence.
    """
    if end <= start:
        raise ValueError('empty window')
    if count <= 0:
        raise ValueError('count must be positive')
    indices = [int(x) for x in np.linspace(start, end - 1, count).round()]
    if indices != sorted(indices) or indices[0] < start or indices[-1] >= end:
        raise AssertionError('invalid uniform indices')
    return indices


def windows(length, window_length=32, stride=32):
    return [(start,min(start+window_length,length)) for start in range(0,length,stride)]


def json_score(text):
    matches=re.findall(r'\{.*?\}',text,re.S)
    if not matches: raise ValueError('no JSON object')
    obj=json.loads(matches[0]); score=obj.get('anomaly_score')
    if isinstance(score,bool) or not isinstance(score,(int,float)): raise ValueError('invalid anomaly_score')
    score=float(score)
    if not 0 <= score <= 100: raise ValueError('score outside [0,100]')
    conf=float(obj.get('confidence',0.0))
    if not 0 <= conf <= 1: raise ValueError('confidence outside [0,1]')
    if obj.get('decision') not in {'normal','uncertain','abnormal'}: raise ValueError('invalid decision')
    return score/100.0,obj


class EndpointScorer:
    def __init__(self, endpoint, model, revision, cache_root, max_tokens=512, timeout=300):
        self.endpoint=endpoint; self.model=model; self.revision=revision; self.cache_root=Path(cache_root)
        self.cache_root.mkdir(parents=True,exist_ok=True); self.max_tokens=max_tokens; self.timeout=timeout
    @staticmethod
    def _data_url(path):
        return 'data:image/jpeg;base64,'+base64.b64encode(Path(path).read_bytes()).decode()
    def score(self, paths, indices):
        key_payload={'model':self.model,'revision':self.revision,'system':DIRECT_SYSTEM,'prompt':DIRECT_PROMPT,
          'frames':[{'index':i,'sha256':sha256_file(p)} for i,p in zip(indices,paths)],'max_tokens':self.max_tokens}
        key=stable_hash(key_payload); saved=self.cache_root/'direct'/f'{key}.json'
        if saved.exists():
            out=json.loads(saved.read_text())
            # A transient endpoint refusal is not a result.  Preserve the
            # diagnostic on disk, but retry and replace it on the next safe
            # resumable invocation rather than making the run permanently fail.
            if out.get('success'):
                out['cache_hit']=True
                return out
        content=[{'type':'text','text':DIRECT_PROMPT}]
        for idx,path in zip(indices,paths):
            content.extend([{'type':'text','text':f'Original frame index: {idx}'}, {'type':'image_url','image_url':{'url':self._data_url(path)}}])
        payload={'model':self.model,'messages':[{'role':'system','content':DIRECT_SYSTEM},{'role':'user','content':content}],
          'temperature':0,'max_tokens':self.max_tokens,'chat_template_kwargs':{'enable_thinking':False},
          'response_format':{'type':'json_object'}}
        t=time.monotonic(); failures=[]
        # A fixed second request handles rare empty/non-JSON generations without
        # changing the prompt, sampling parameters, frames, or score protocol.
        for attempt in range(2):
            raw_text=None
            try:
                req=urllib.request.Request(self.endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer EMPTY'})
                with urllib.request.urlopen(req,timeout=self.timeout) as resp: body=json.loads(resp.read())
                raw_text=body['choices'][0]['message']['content'].strip(); score,parsed=json_score(raw_text)
                out={'success':True,'score':score,'parsed':parsed,'raw_response':raw_text,'usage':body.get('usage',{}),
                     'elapsed_seconds':time.monotonic()-t,'cache_hit':False,'key':key,'attempt':attempt+1,'failures_before_success':failures}
                break
            except Exception as exc:
                failures.append({'attempt':attempt+1,'error':repr(exc),'raw_response':raw_text})
        else:
            out={'success':False,'error':failures[-1]['error'],'attempts':failures,
                 'elapsed_seconds':time.monotonic()-t,'cache_hit':False,'key':key}
        atomic_json(saved,out); return out


def run_direct(workspace, run_root, setup, scorer, *, window_length=32, stride=32, sample_count=4, video_ids=None, smoke=False, method='Direct VLM', subdir='direct', extra_config=None):
    if setup not in SETUPS: raise ValueError(setup)
    source,target=SETUPS[setup]; data=ShanghaiData(workspace)
    out=Path(run_root)/subdir/('smoke' if smoke else 'setups')/setup; out.mkdir(parents=True,exist_ok=True)
    if (out/'prediction_lock.json').exists() and (out/'frame_scores.npy').exists(): return out
    manifest=data.target_manifest(target)
    if video_ids:
        wanted=set(video_ids); manifest=[x for x in manifest if x['video_id'] in wanted]
        if len(manifest)!=len(wanted): raise ValueError('requested video absent from target')
    sanitized=[{'video_id':x['video_id'],'frame_count':x['frame_count'],'prediction_length':x['gt_end']-x['gt_start']} for x in manifest]
    config={'method':method,'setup':setup,'source_domain':source,'target_domain':target,'model':scorer.model,'revision':scorer.revision,
      'endpoint':scorer.endpoint,'prompt':DIRECT_PROMPT,'prompt_sha256':hashlib.sha256((DIRECT_SYSTEM+DIRECT_PROMPT).encode()).hexdigest(),
      'sampling':{'method':'uniform_temporal','window_length':window_length,'stride':stride,'sample_count':sample_count},'seed':2022,
      'inference':{'temperature':0,'max_new_tokens':getattr(scorer,'max_tokens',getattr(scorer,'max_new_tokens',None))},
      'target_manifest':sanitized,'gt_read':False,'smoke':smoke,**(extra_config or {})}
    atomic_json(out/'config.json',config); atomic_json(out/'inference_config.json',config)
    (out/'target_test_video_ids.txt').write_text('\n'.join(x['video_id'] for x in sanitized)+'\n')
    all_scores=[]; videos=[]
    for item in manifest:
        vid=item['video_id']; paths=data.test_frames(vid); values=np.empty(len(paths),dtype=np.float32); records=[]
        for start,end in windows(len(paths),window_length,stride):
            idx=uniform_indices(start,end,sample_count); result=scorer.score([paths[i] for i in idx],idx)
            if not result['success']: raise RuntimeError(f'{vid} [{start}:{end}]: {result}')
            values[start:end]=result['score']; record={'video_id':vid,'start':start,'end':end,'sample_indices':idx,**result}
            records.append(record); append_jsonl(out/'raw_vlm_responses.jsonl',record)
        aligned=pad_last(values,item['gt_end']-item['gt_start']).astype(np.float32); np.save(out/f'{vid}_frame_scores.npy',aligned)
        videos.append({'video_id':vid,'raw_frame_count':len(paths),'prediction_length':len(aligned),'windows':len(records),'cache_hits':sum(x['cache_hit'] for x in records)})
        all_scores.extend(aligned.tolist())
    scores=np.asarray(all_scores,dtype=np.float32)
    if not np.isfinite(scores).all() or not ((scores>=0)&(scores<=1)).all(): raise ValueError('invalid scores')
    np.save(out/'frame_scores.npy',scores)
    atomic_json(out/'inference_manifest.json',{'videos':videos,'prediction_length':len(scores),'frame_scores_sha256':sha256_file(out/'frame_scores.npy'),'gt_read':False})
    atomic_json(out/'prediction_lock.json',{'setup':setup,'prediction':str(out/'frame_scores.npy'),'sha256':sha256_file(out/'frame_scores.npy'),'length':len(scores),'prompt_sha256':config['prompt_sha256'],'gt_read_before_lock':False})
    return out


ZERO_SHOT_TARGETS = {'domain_01': '01', 'domain_08': '08', 'mixed': 'mixed'}


def run_zeroshot_target(workspace, run_root, target_name, scorer, *, window_length=240,
                         sample_count=8, video_ids=None, smoke=False):
    """Run a strict target-only Direct-VLM prediction pass.

    This function deliberately has no source domain, train-data access, rule
    access, or evaluator import.  GT is read only by the separate locked-score
    evaluator after this function has created ``prediction_lock.json``.
    """
    if target_name not in ZERO_SHOT_TARGETS:
        raise ValueError(f'unknown target {target_name!r}')
    if sample_count != 8:
        raise ValueError('zero-shot protocol is fixed at exactly 8 frames/window')
    if window_length <= 0:
        raise ValueError('window_length must be positive')
    data = ShanghaiData(workspace)
    target = ZERO_SHOT_TARGETS[target_name]
    out = Path(run_root) / ('smoke' if smoke else 'targets') / target_name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'prediction_lock.json').exists() and (out/'frame_scores.npy').exists():
        return out
    manifest = data.target_manifest(target)
    if video_ids:
        wanted = set(video_ids)
        manifest = [x for x in manifest if x['video_id'] in wanted]
        if len(manifest) != len(wanted):
            raise ValueError('requested video absent from target')
    if not manifest:
        raise ValueError('target selection is empty')
    sanitized = [{'video_id': x['video_id'], 'frame_count': x['frame_count'],
                  'prediction_length': x['gt_end']-x['gt_start']} for x in manifest]
    prompt_hash = hashlib.sha256((DIRECT_SYSTEM+DIRECT_PROMPT).encode()).hexdigest()
    config = {
        'method': 'Direct VLM zero-shot target-only',
        'target_name': target_name, 'target_domain': target,
        'source_data_used': False, 'source_domain': None,
        'model': scorer.model, 'revision': scorer.revision, 'endpoint': scorer.endpoint,
        'prompt': DIRECT_PROMPT, 'prompt_sha256': prompt_hash,
        'sampling': {'method': 'fixed_nonoverlap_uniform_temporal',
                     'window_frames': window_length, 'stride_frames': window_length,
                     'nominal_fps': 24, 'nominal_window_seconds': window_length/24,
                     'sample_count': sample_count, 'tail_policy': 'retain_and_repeat_to_8'},
        'seed': 2022, 'inference': {'temperature': 0,
                                     'max_new_tokens': getattr(scorer, 'max_tokens', 512),
                                     'enable_thinking': False},
        'target_manifest': sanitized, 'gt_read': False, 'smoke': smoke,
    }
    atomic_json(out/'config.json', config)
    atomic_json(out/'inference_config.json', config)
    (out/'target_test_video_ids.txt').write_text('\n'.join(x['video_id'] for x in sanitized)+'\n')
    all_scores, videos = [], []
    for item in manifest:
        vid = item['video_id']; paths = data.test_frames(vid)
        per_video_path = out/f'{vid}_frame_scores.npy'
        expected = item['gt_end']-item['gt_start']
        if per_video_path.exists():
            aligned = np.load(per_video_path)
            if aligned.shape == (expected,) and np.isfinite(aligned).all() and ((aligned >= 0) & (aligned <= 1)).all():
                all_scores.extend(aligned.astype(np.float32).tolist())
                videos.append({'video_id': vid, 'raw_frame_count': len(paths), 'prediction_length': len(aligned),
                               'windows': len(windows(len(paths), window_length, window_length)), 'resumed': True})
                continue
        values = np.empty(len(paths), dtype=np.float32); records = []
        for start, end in windows(len(paths), window_length, window_length):
            indices = uniform_indices_fixed_count(start, end, sample_count)
            result = scorer.score([paths[i] for i in indices], indices)
            if not result['success']:
                raise RuntimeError(f'{vid} [{start}:{end}]: {result}')
            values[start:end] = result['score']
            record = {'video_id': vid, 'start_frame': start, 'end_frame_exclusive': end,
                      'valid_frames': end-start, 'sample_indices': indices,
                      'sample_count': len(indices), **result}
            records.append(record)
            append_jsonl(out/'window_manifest.jsonl', record)
            append_jsonl(out/'raw_vlm_responses.jsonl', record)
        aligned = pad_last(values, expected).astype(np.float32)
        if len(aligned) != expected:
            raise AssertionError(f'{vid}: reconstruction length mismatch')
        np.save(per_video_path, aligned)
        all_scores.extend(aligned.tolist())
        videos.append({'video_id': vid, 'raw_frame_count': len(paths), 'prediction_length': len(aligned),
                       'windows': len(records), 'resumed': False,
                       'cache_hits': sum(x['cache_hit'] for x in records)})
    scores = np.asarray(all_scores, dtype=np.float32)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('invalid scores')
    np.save(out/'frame_scores.npy', scores)
    atomic_json(out/'inference_manifest.json', {'videos': videos, 'prediction_length': len(scores),
                'frame_scores_sha256': sha256_file(out/'frame_scores.npy'), 'gt_read': False,
                'target_order': [x['video_id'] for x in sanitized]})
    atomic_json(out/'prediction_lock.json', {'target_name': target_name,
                'prediction': str(out/'frame_scores.npy'), 'sha256': sha256_file(out/'frame_scores.npy'),
                'length': len(scores), 'prompt_sha256': prompt_hash, 'gt_read_before_lock': False})
    return out
