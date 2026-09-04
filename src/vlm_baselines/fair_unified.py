"""Fair Direct-VLM primitives: one Transformers path, adapter optional."""
from __future__ import annotations
import hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from anomaly_ruler.core import atomic_json, append_jsonl, natural_key, sha256_file
from anomaly_ruler.data import ShanghaiData
from .dense_zeroshot import SYSTEM, PROMPT, temporal_centers, context_bounds
from .common import uniform_indices_fixed_count

FAIR = {'context_frames':240,'center_stride':16,'sample_count':8,
        'min_pixels':56*56,'max_pixels':336*336,'seed':2022}
NATIVE_RESOLUTION = {'min_pixels':65536, 'max_pixels':16777216}


def fair_config(*, native_resolution=False):
    """Return the fixed Direct-VLM protocol with an explicit pixel policy."""
    cfg = dict(FAIR)
    if native_resolution:
        cfg.update(NATIVE_RESOLUTION)
    return cfg


class WeakShanghaiTrainData:
    """Source-only view of the official ShanghaiTech WVAD training split.

    This deliberately does not open the test manifest, a frame mask, or a GT
    file.  The abnormal WVAD training videos happen to be physically stored in
    ``testing/frames`` in this normalized layout, but their membership and weak
    labels come *only* from ``SH_Train.txt``.
    """
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        train = (self.workspace/'splits/shanghaitech_wvad/SH_Train.txt').read_text().splitlines()
        ids = [Path(x.strip()).stem for x in train if x.strip()]
        if len(ids) != 238:
            raise ValueError(f'expected 238 weak-training IDs, got {len(ids)}')
        self.train_abnormal, self.train_normal = ids[:63], ids[63:]
        if len(self.train_abnormal) != 63 or len(self.train_normal) != 175:
            raise AssertionError('unexpected official weak-training class counts')

    @staticmethod
    def _frames(root):
        paths = sorted([*root.glob('*.jpg'), *root.glob('*.jpeg'), *root.glob('*.png')], key=natural_key)
        if not paths:
            raise FileNotFoundError(f'No frames under {root}')
        return paths

    def weak_train_frames(self, video_id):
        if video_id in self.train_normal:
            return self._frames(self.workspace/'work/shanghaitech_frames/normal'/video_id)
        if video_id in self.train_abnormal:
            return self._frames(self.workspace/'datasets/shanghaitech/testing/frames'/video_id)
        raise KeyError(f'{video_id} absent from SH_Train.txt')

def processor_inputs(processor, paths, indices, cfg=FAIR, *, prompt=PROMPT, system=SYSTEM):
    from PIL import Image
    imgs=[Image.open(p).convert('RGB') for p in paths]
    content=[{'type':'text','text':prompt}]
    for i,img in zip(indices,imgs):
        content += [{'type':'text','text':f'Original frame index: {i}'},{'type':'image','image':img}]
    messages=[{'role':'system','content':system},{'role':'user','content':content}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return processor(text=[text],images=imgs,return_tensors='pt',padding=True,
                     min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'])

def peft_config():
    from peft import LoraConfig, TaskType
    return LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',
                      target_modules=['q_proj','k_proj','v_proj','o_proj'],
                      task_type=TaskType.CAUSAL_LM)

class UnifiedScorer:
    """The exact same scorer for Base Qwen and a PEFT adapter."""
    def __init__(self, model_path, *, adapter_path=None, device='cuda:0', cfg=None, trainable=False,
                 prompt=PROMPT, system_prompt=SYSTEM):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.cfg=dict(FAIR if cfg is None else cfg); self.device=device
        self.model='Qwen/Qwen3.8-27B'; self.revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
        self.endpoint='local_transformers'
        self.prompt, self.system_prompt = prompt, system_prompt
        self.processor=AutoProcessor.from_pretrained(model_path)
        net=AutoModelForImageTextToText.from_pretrained(model_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device)
        net.config.use_cache=False
        if adapter_path:
            from peft import PeftModel
            net=PeftModel.from_pretrained(net,adapter_path,is_trainable=trainable)
        elif trainable:
            from peft import get_peft_model
            net=get_peft_model(net,peft_config())
        else:
            for p in net.parameters(): p.requires_grad=False
        self.net=net
        if not trainable:
            self.net.eval()
        tok=self.processor.tokenizer; self.id0=tok.encode('0',add_special_tokens=False); self.id1=tok.encode('1',add_special_tokens=False)
        if len(self.id0)!=1 or len(self.id1)!=1: raise ValueError('0/1 are not single tokens')
        self.id0,self.id1=self.id0[0],self.id1[0]
        self.adapter_path=adapter_path
    def logit(self,paths,indices):
        x={k:v.to(self.device) if hasattr(v,'to') else v for k,v in processor_inputs(
            self.processor, paths, indices, self.cfg, prompt=self.prompt, system=self.system_prompt).items()}
        o=self.net(**x); last=int(x['attention_mask'][0].sum())-1; z=o.logits[0,last]
        return z[self.id1]-z[self.id0]
    def score(self,paths,indices):
        started=time.monotonic()
        with torch.no_grad(): z=self.logit(paths,indices); p=float(torch.sigmoid(z).cpu())
        return {'success':True,'score':p,'logit':float(z.cpu()),'forced_choice':'1' if p>=.5 else '0',
                'cache_hit':False, 'elapsed_seconds':time.monotonic()-started}

def source_split(data,source,seed=2022,val_fraction=.2):
    a=[v for v in data.train_abnormal if source=='mixed' or v.split('_')[0]==source]; n=[v for v in data.train_normal if source=='mixed' or v.split('_')[0]==source]
    rng=random.Random(seed);rng.shuffle(a);rng.shuffle(n); va=max(1,round(len(a)*val_fraction));vn=max(1,round(len(n)*val_fraction))
    return {'train_abnormal':a[va:],'train_normal':n[vn:],'val_abnormal':a[:va],'val_normal':n[:vn]}

def epoch_records(data,video_id,epoch,limit=8,cfg=FAIR):
    p=data.weak_train_frames(video_id); cs=temporal_centers(len(p),cfg['center_stride']); rng=np.random.default_rng(cfg['seed']+epoch+int(hashlib.sha256(video_id.encode()).hexdigest()[:8],16)); take=sorted(rng.choice(len(cs),size=min(limit,len(cs)),replace=False).tolist())
    return [( [str(p[i]) for i in uniform_indices_fixed_count(*context_bounds(cs[j],len(p),cfg['context_frames']),cfg['sample_count'])], uniform_indices_fixed_count(*context_bounds(cs[j],len(p),cfg['context_frames']),cfg['sample_count'])) for j in take]

def streaming_mil_step(scorer, records, abnormal, topk=2, gradient_scale=1.0):
    """Exact detached top-k MIL, with only one differentiable clip graph alive."""
    with torch.no_grad(): probe=[float(scorer.logit(p,i).cpu()) for p,i in records]
    chosen=np.argsort(probe)[-min(topk,len(probe)):].tolist() if abnormal else list(range(len(records)))
    total=0.0
    for j in chosen:
        z=scorer.logit(*records[j]); loss=(torch.nn.functional.softplus(-z) if abnormal else torch.nn.functional.softplus(z))/len(chosen)
        (loss/gradient_scale).backward(); total += float(loss.detach())
    return total,probe,chosen

def train_peft_mil(workspace, run_root, source, model_path, *, epochs=3, lr=1e-4,
                   topk=2, clips=8, seed=2022, cfg=None, grad_accumulation=4,
                   max_train_videos_per_class=None, max_val_videos_per_class=None):
    """Source-only video-bag MIL training; no mask/GT import or access."""
    cfg = dict(FAIR if cfg is None else cfg)
    if grad_accumulation < 1:
        raise ValueError('grad_accumulation must be >= 1')
    out=Path(run_root)/'adapters'/f'LoRA_{source}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'training_complete.json').exists(): return out
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed); data=WeakShanghaiTrainData(workspace); full_split=source_split(data,source,seed)
    split={k:list(v) for k,v in full_split.items()}
    if max_train_videos_per_class is not None:
        split['train_abnormal']=split['train_abnormal'][:max_train_videos_per_class]
        split['train_normal']=split['train_normal'][:max_train_videos_per_class]
    if max_val_videos_per_class is not None:
        split['val_abnormal']=split['val_abnormal'][:max_val_videos_per_class]
        split['val_normal']=split['val_normal'][:max_val_videos_per_class]
    if not split['train_abnormal'] or not split['train_normal'] or not split['val_abnormal'] or not split['val_normal']:
        raise ValueError(f'{source}: source training split must contain both weak classes')
    atomic_json(out/'train_config.json',{
        'method':'Weakly-Supervised PEFT LoRA Direct VLM', 'source':source, 'split':split,
        'full_source_split':full_split,
        'fair_config':cfg, 'prompt':PROMPT, 'system_prompt':SYSTEM,
        'prompt_sha256':hashlib.sha256((SYSTEM+PROMPT).encode()).hexdigest(),
        'lora':{'implementation':'PEFT', 'r':16,'alpha':32,'dropout':.05,
                'targets':['q_proj','k_proj','v_proj','o_proj'], 'vision_encoder_frozen':True,
                'base_language_model_frozen':True},
        'epochs':epochs,'lr':lr,'topk':topk,'clips_per_video':clips,
        'batch_size_videos':1, 'gradient_accumulation':grad_accumulation, 'seed':seed,
        'max_train_videos_per_class':max_train_videos_per_class,
        'max_val_videos_per_class':max_val_videos_per_class,
        'frame_masks_used':False,'target_gt_used':False,
        'training_data':'SH_Train.txt video-level weak labels only',
        'test_manifest_read':False,
    })
    for k,v in split.items():(out/(k+'.txt')).write_text('\n'.join(v)+'\n')
    s=UnifiedScorer(model_path,cfg=cfg,trainable=True);s.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    param_rows=[{'name':n,'parameters':p.numel(),'requires_grad':p.requires_grad} for n,p in s.net.named_parameters()]
    bad=[x['name'] for x in param_rows if x['requires_grad'] and 'lora_' not in x['name']]
    if bad:
        raise RuntimeError(f'non-LoRA parameter unexpectedly trainable: {bad[:5]}')
    trainable=sum(x['parameters'] for x in param_rows if x['requires_grad']); total=sum(x['parameters'] for x in param_rows)
    if trainable == 0:
        raise RuntimeError('no PEFT LoRA parameters are trainable')
    atomic_json(out/'trainable_parameters.json',{
        'total_parameters':total, 'trainable_parameters':trainable,
        'trainable_ratio':trainable/total, 'trainable_parameter_names':[x['name'] for x in param_rows if x['requires_grad']],
        'all_trainable_parameters_are_lora':True,
    })
    opt=torch.optim.AdamW((p for p in s.net.parameters() if p.requires_grad),lr=lr); best=float('inf')
    train=[(v,1) for v in split['train_abnormal']]+[(v,0) for v in split['train_normal']]
    for ep in range(1,epochs+1):
        random.Random(seed+ep).shuffle(train);s.net.train(); losses=[]; opt.zero_grad(set_to_none=True)
        for step,(vid,y) in enumerate(train,1):
            loss,probe,chosen=streaming_mil_step(s,epoch_records(data,vid,ep,clips,cfg),bool(y),topk,grad_accumulation)
            if step % grad_accumulation == 0 or step == len(train):
                opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(loss);append_jsonl(out/'training_log.jsonl',{'epoch':ep,'step':step,'video_id':vid,'weak_video_label':y,'loss':loss,'topk_indices':chosen,'optimizer_step':step % grad_accumulation == 0 or step == len(train)})
        s.net.eval(); vals=[]
        with torch.no_grad():
            for y,k in [(1,'val_abnormal'),(0,'val_normal')]:
                for vid in split[k]:
                    z=torch.tensor([float(s.logit(*r).cpu()) for r in epoch_records(data,vid,ep,clips,cfg)]);q=z.topk(min(topk,len(z))).values if y else z;vals.append(float((torch.nn.functional.softplus(-q) if y else torch.nn.functional.softplus(q)).mean()))
        val=float(np.mean(vals));atomic_json(out/f'epoch_{ep:02d}.json',{'train_loss':float(np.mean(losses)),'val_mil_loss':val})
        if val<best: best=val;s.net.save_pretrained(out/'selected_adapter');s.processor.save_pretrained(out/'processor')
    atomic_json(out/'training_complete.json',{'selected_checkpoint':'selected_adapter','selected_val_mil_loss':best,'source_only':True});return out
