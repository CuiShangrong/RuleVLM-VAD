from __future__ import annotations

import json
import os
import random
import hashlib
from pathlib import Path

import numpy as np

from anomaly_ruler.core import append_jsonl, atomic_json, sha256_file
from anomaly_ruler.data import SETUPS, ShanghaiData
from .common import DIRECT_PROMPT, DIRECT_SYSTEM, json_score, uniform_indices

TUNING_PROMPT = DIRECT_PROMPT


def source_weak_clips(workspace, source, *, clips_per_video=2, clip_length=32, sample_count=4, limit_videos=None, seed=2022):
    data=ShanghaiData(workspace)
    abnormal=[x for x in data.train_abnormal if source=='mixed' or x.split('_')[0]==source]
    normal=[x for x in data.train_normal if source=='mixed' or x.split('_')[0]==source]
    records=[]
    for label,ids in [(1,abnormal),(0,normal)]:
        selected=ids[:limit_videos] if limit_videos else ids
        for vid in selected:
            frames=data.weak_train_frames(vid); starts=np.linspace(0,max(0,len(frames)-1),clips_per_video).round().astype(int).tolist()
            for clip_index,start in enumerate(starts):
                end=min(start+clip_length,len(frames)); indices=uniform_indices(start,end,sample_count)
                records.append({'video_id':vid,'video_label':label,'clip_index':clip_index,'start':start,'end':end,
                  'sample_indices':indices,'frame_paths':[str(frames[i]) for i in indices]})
    rng=random.Random(seed); rng.shuffle(records)
    return records,{'abnormal_videos':abnormal,'normal_videos':normal}


def _load_inputs(processor, image_paths, target=None):
    from PIL import Image
    images=[Image.open(p).convert('RGB') for p in image_paths]
    user=[{'type':'text','text':TUNING_PROMPT}]+[{'type':'image','image':im} for im in images]
    base=[{'role':'system','content':DIRECT_SYSTEM},{'role':'user','content':user}]
    if target is None:
        text=processor.apply_chat_template(base,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=processor(text=[text],images=images,return_tensors='pt',padding=True)
        return inputs
    full=base+[{'role':'assistant','content':target}]
    full_text=processor.apply_chat_template(full,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    prompt_text=processor.apply_chat_template(base,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=processor(text=[full_text],images=images,return_tensors='pt',padding=True)
    prompt_ids=processor(text=[prompt_text],return_tensors='pt',padding=True)['input_ids']
    labels=inputs['input_ids'].clone(); labels[:,:prompt_ids.shape[1]]=-100
    labels[labels==getattr(processor.tokenizer,'pad_token_id',-999999)]=-100
    inputs['labels']=labels
    return inputs


def train_lora(workspace, run_root, setup, model_path, *, device='cuda:0', steps=100, learning_rate=2e-4, rank=8, alpha=16, clips_per_video=2, clip_length=32, sample_count=4, limit_videos=None, seed=2022, adapter_name=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, TaskType, get_peft_model
    if setup not in SETUPS: raise ValueError(setup)
    source,target=SETUPS[setup]
    out=Path(run_root)/'tuned'/'adapters'/(adapter_name or f'{setup}_lora_v1_no_thinking'); out.mkdir(parents=True,exist_ok=True)
    final=out/'adapter_config.json'
    if final.exists(): return out
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    records,ids=source_weak_clips(workspace,source,clips_per_video=clips_per_video,clip_length=clip_length,sample_count=sample_count,limit_videos=limit_videos,seed=seed)
    if not records or not ids['abnormal_videos'] or not ids['normal_videos']: raise ValueError('source must contain both weak labels')
    target_ids=[x['video_id'] for x in ShanghaiData(workspace).target_manifest(target)]
    overlap=sorted(set(ids['abnormal_videos']+ids['normal_videos']) & set(target_ids))
    if overlap: raise ValueError(f'source/target video leakage: {overlap}')
    atomic_json(out/'train_config.json',{'method':'Tuned VLM LoRA weak supervision','setup':setup,'source_domain':source,'target_domain':target,
      'model_path':model_path,'model_sha256_config':sha256_file(Path(model_path)/'config.json'),'device':device,'seed':seed,'steps':steps,'learning_rate':learning_rate,
      'lora':{'rank':rank,'alpha':alpha,'dropout':0.0,'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']},
      'sampling':{'uniform_temporal':True,'clip_length':clip_length,'sample_count':sample_count,'clips_per_video':clips_per_video},'enable_thinking':False,
      'supervision':'weak video-level label copied to sampled clips; no frame-level source label','source_ids':ids,'target_test_video_ids':target_ids,
      'source_target_video_overlap':overlap,'source_target_leakage_check':'PASSED','target_gt_used':False})
    (out/'source_train_video_ids.txt').write_text('\n'.join(ids['abnormal_videos']+ids['normal_videos'])+'\n')
    with (out/'train_samples.jsonl').open('w') as f:
        for x in records: f.write(json.dumps(x)+'\n')
    processor=AutoProcessor.from_pretrained(model_path)
    model=AutoModelForImageTextToText.from_pretrained(model_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device)
    model.config.use_cache=False
    if hasattr(model,'gradient_checkpointing_enable'): model.gradient_checkpointing_enable()
    lora=LoraConfig(task_type=TaskType.CAUSAL_LM,r=rank,lora_alpha=alpha,lora_dropout=0.0, bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model=get_peft_model(model,lora); model.train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=learning_rate)
    log=out/'training_log.jsonl'
    for step in range(steps):
        rec=records[step%len(records)]; target_text=json.dumps({'anomaly_score':100 if rec['video_label'] else 0,'confidence':1.0,'decision':'abnormal' if rec['video_label'] else 'normal'},separators=(',',':'))
        inputs=_load_inputs(processor,rec['frame_paths'],target_text)
        inputs={k:v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
        optimizer.zero_grad(set_to_none=True); loss=model(**inputs).loss
        if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss at {step}')
        loss.backward(); optimizer.step()
        append_jsonl(log,{'step':step+1,'loss':float(loss.detach().cpu()),'video_id':rec['video_id'],'video_label':rec['video_label']})
    model.save_pretrained(out); processor.save_pretrained(out/'processor')
    atomic_json(out/'training_complete.json',{'steps':steps,'adapter_config_sha256':sha256_file(out/'adapter_config.json'),'final_loss':float(loss.detach().cpu()),'source_only_check':True})
    del model; torch.cuda.empty_cache(); return out


class TunedTransformerScorer:
    def __init__(self, model_path, adapter_path, *, device='cuda:0', max_new_tokens=512):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
        self.torch=torch; self.device=device; self.model_path=model_path; self.adapter_path=adapter_path; self.model='Qwen/Qwen3.8-27B'
        self.revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'; self.endpoint=f'local_transformers:{device}'; self.cache_root=Path(adapter_path).parent.parent/'inference_cache'; self.cache_root.mkdir(parents=True,exist_ok=True); self.max_new_tokens=max_new_tokens
        self.processor=AutoProcessor.from_pretrained(adapter_path+'/processor')
        base=AutoModelForImageTextToText.from_pretrained(model_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device)
        self.net=PeftModel.from_pretrained(base,adapter_path).eval()
    def score(self, paths, indices):
        key=hashlib.sha256((f'score_extract_v4_no_thinking_max{self.max_new_tokens}_'+sha256_file(Path(self.adapter_path)/'adapter_config.json')+'_'+'_'.join(sha256_file(p) for p in paths)).encode()).hexdigest()
        cache=self.cache_root/(key+'.json')
        if cache.exists(): out=json.loads(cache.read_text());out['cache_hit']=True;return out
        text=None
        try:
            inputs=_load_inputs(self.processor,paths); inputs={k:v.to(self.device) if hasattr(v,'to') else v for k,v in inputs.items()}
            with self.torch.no_grad(): output=self.net.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
            prompt_len=inputs['input_ids'].shape[1]; text=self.processor.decode(output[0][prompt_len:],skip_special_tokens=True).strip(); score,parsed=json_score(text)
            out={'success':True,'score':score,'parsed':parsed,'raw_response':text,'cache_hit':False}
        except Exception as exc: out={'success':False,'error':repr(exc),'raw_response':text,'cache_hit':False}
        atomic_json(cache,out); return out
