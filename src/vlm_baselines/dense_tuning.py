"""Source-only weak MIL LoRA tuning for dense forced-choice Qwen VAD."""
from __future__ import annotations
import hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from anomaly_ruler.core import atomic_json,append_jsonl,sha256_file
from anomaly_ruler.data import ShanghaiData
from .dense_zeroshot import SYSTEM,PROMPT,temporal_centers,context_bounds,TokenProbabilityScorer
from .common import uniform_indices_fixed_count

# Eight raw 1080p frames otherwise expand to an impractical number of visual
# tokens for a single-GPU training forward.  This is image preprocessing only:
# the dense temporal context, 8 samples, prompt, forced 0/1 logits and MIL
# protocol remain fixed.  Both training and tuned inference call `_inputs`.
VISION_MIN_PIXELS = 56 * 56
VISION_MAX_PIXELS = 112 * 112

class NativeLoRALinear(nn.Module):
    def __init__(self,base,r,alpha):
        super().__init__();self.base=base;self.r=r;self.scaling=alpha/r
        # Injection happens after the frozen base has been placed on CUDA.  New
        # Parameters otherwise default to CPU and fail on the first forward.
        kwargs={'device':base.weight.device,'dtype':base.weight.dtype}
        self.A=nn.Parameter(torch.zeros(r,base.in_features,**kwargs));self.B=nn.Parameter(torch.zeros(base.out_features,r,**kwargs));nn.init.kaiming_uniform_(self.A,a=5**.5)
        for p in self.base.parameters():p.requires_grad=False
    def forward(self,x): return self.base(x)+(x@self.A.T@self.B.T)*self.scaling
def inject_native_lora(model,r,alpha,targets=('q_proj','k_proj','v_proj','o_proj')):
    replaced=[]
    for name,module in list(model.named_modules()):
        if name.split('.')[-1] not in targets or not isinstance(module,nn.Linear):continue
        parent=model
        for p in name.split('.')[:-1]:parent=getattr(parent,p)
        setattr(parent,name.split('.')[-1],NativeLoRALinear(module,r,alpha));replaced.append(name)
    if not replaced:raise ValueError('no LoRA target modules found')
    return replaced
def lora_state(model):return {n:p.detach().cpu() for n,p in model.named_parameters() if '.A' in n or '.B' in n}

def source_ids(data,source):
    a=[x for x in data.train_abnormal if source=='mixed' or x.split('_')[0]==source]
    n=[x for x in data.train_normal if source=='mixed' or x.split('_')[0]==source]
    if not a or not n: raise ValueError('source requires normal and abnormal videos')
    return a,n
def stratify(a,n,seed,val_fraction=.2):
    rng=random.Random(seed); a=list(a);n=list(n);rng.shuffle(a);rng.shuffle(n)
    va=max(1,round(len(a)*val_fraction));vn=max(1,round(len(n)*val_fraction))
    return {'train_abnormal':a[va:],'train_normal':n[vn:],'val_abnormal':a[:va],'val_normal':n[:vn]}
def clip_records(data,vid,label,centers_per_video=8,context=240):
    paths=data.weak_train_frames(vid); allc=temporal_centers(len(paths),16); take=np.linspace(0,len(allc)-1,min(centers_per_video,len(allc))).round().astype(int)
    out=[]
    for j in sorted(set(take.tolist())):
        c=allc[j];lo,hi=context_bounds(c,len(paths),context);ix=uniform_indices_fixed_count(lo,hi,8)
        out.append({'video_id':vid,'video_label':label,'center_frame':c,'context_start':lo,'context_end_exclusive':hi,'sample_indices':ix,'frame_paths':[str(paths[i]) for i in ix]})
    return out
def _inputs(processor,paths):
    from PIL import Image
    imgs=[Image.open(p).convert('RGB') for p in paths]
    content=[{'type':'text','text':PROMPT}]
    for i,x in enumerate(imgs): content += [{'type':'text','text':f'Frame order {i}'},{'type':'image','image':x}]
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':content}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return processor(text=[text],images=imgs,return_tensors='pt',padding=True,
                     min_pixels=VISION_MIN_PIXELS,max_pixels=VISION_MAX_PIXELS)
def _binary_logit(model,inputs,id0,id1,device):
    x={k:v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()};o=model(**x)
    last=int(x['attention_mask'][0].sum().item())-1; z=o.logits[0,last]
    return z[id1]-z[id0]
def _video_loss(model,processor,rows,id0,id1,device,topk,normal):
    logits=[_binary_logit(model,_inputs(processor,r['frame_paths']),id0,id1,device) for r in rows]
    z=torch.stack(logits)
    if normal: return torch.nn.functional.softplus(z).mean(),z
    k=min(topk,len(rows)); ix=torch.topk(z.detach(),k).indices
    return torch.nn.functional.softplus(-z[ix]).mean(),z

def train_dense_lora(workspace,run_root,source,model_path,*,device='cuda:0',epochs=3,rank=4,alpha=8,lr=1e-4,centers_per_video=8,topk=2,seed=2022,val_fraction=.2,smoke=False):
    from transformers import AutoModelForImageTextToText,AutoProcessor
    out=Path(run_root)/'adapters'/f'LoRA_{source}';out.mkdir(parents=True,exist_ok=True)
    if (out/'training_complete.json').exists(): return out
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);data=ShanghaiData(workspace);a,n=source_ids(data,source);split=stratify(a,n,seed,val_fraction)
    # A smoke run is intentionally tiny but still contains both weak classes
    # and source-only validation; it is never used as a formal checkpoint.
    if smoke: split={k:v[:1] for k,v in split.items()}
    atomic_json(out/'train_config.json',{'method':'Dense Qwen3.8-27B native LoRA Direct VLM weak MIL','source_domain':source,'model':'Qwen/Qwen3.8-27B','model_path':model_path,'model_config_sha256':sha256_file(Path(model_path)/'config.json'),'seed':seed,'epochs':epochs,'lr':lr,'lora':{'implementation':'native_frozen_linear_low_rank','rank':rank,'alpha':alpha,'targets':['q_proj','k_proj','v_proj','o_proj']},'sampling':{'context_frames':240,'center_stride':16,'sample_count':8,'centers_per_video':centers_per_video},'image_preprocessing':{'min_pixels':VISION_MIN_PIXELS,'max_pixels':VISION_MAX_PIXELS,'reason':'fixed visual token budget for 8-frame single-GPU Qwen training'},'mil':{'normal':'all sampled clips BCE-to-normal','abnormal':f'top-{topk} sampled clips BCE-to-abnormal','no_frame_masks':True},'split':split,'target_gt_used':False,'smoke':smoke})
    for k,v in split.items():(out/(k+'.txt')).write_text('\n'.join(v)+'\n')
    proc=AutoProcessor.from_pretrained(model_path);tok=proc.tokenizer;id0=tok.encode('0',add_special_tokens=False);id1=tok.encode('1',add_special_tokens=False)
    if len(id0)!=1 or len(id1)!=1: raise ValueError('0/1 must be single tokenizer tokens')
    base=AutoModelForImageTextToText.from_pretrained(model_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device);base.config.use_cache=False
    for p in base.parameters():p.requires_grad=False
    replaced=inject_native_lora(base,rank,alpha)
    # LoRA gradients need layer activations, not frozen base-weight gradients.
    # Checkpointing recomputes these activations during backward and keeps the
    # single-GPU BF16 footprint bounded for eight-frame clips.
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=base;atomic_json(out/'adapter_config.json',{'rank':rank,'alpha':alpha,'targets':['q_proj','k_proj','v_proj','o_proj'],'gradient_checkpointing':True,'reentrant':False,'replaced_modules':replaced})
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr)
    train=[(x,1) for x in split['train_abnormal']]+[(x,0) for x in split['train_normal']];best=float('inf');log=out/'training_log.jsonl'
    for ep in range(1,epochs+1):
        random.Random(seed+ep).shuffle(train);model.train();losses=[]
        for step,(vid,y) in enumerate(train,1):
            rows=clip_records(data,vid,y,centers_per_video);loss,_=_video_loss(model,proc,rows,id0[0],id1[0],device,topk,y==0);opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()));append_jsonl(log,{'epoch':ep,'step':step,'video_id':vid,'weak_video_label':y,'loss':losses[-1]})
        model.eval();vl=[]
        with torch.no_grad():
            for y,key in [(1,'val_abnormal'),(0,'val_normal')]:
                for vid in split[key]:
                    rows=clip_records(data,vid,y,centers_per_video);z=[]
                    for r in rows:z.append(_binary_logit(model,_inputs(proc,r['frame_paths']),id0[0],id1[0],device))
                    zz=torch.stack(z);vl.append(float((torch.nn.functional.softplus(-zz.topk(min(topk,len(zz))).values).mean() if y else torch.nn.functional.softplus(zz).mean()).cpu()))
        val=float(np.mean(vl)); ck=out/f'checkpoint_epoch_{ep:02d}.pt';torch.save(lora_state(model),ck);atomic_json(out/f'epoch_{ep:02d}.json',{'train_loss':float(np.mean(losses)),'val_mil_loss':val,'checkpoint':str(ck)})
        if val<best:
            best=val;(out/'selected_adapter').mkdir(exist_ok=True);torch.save(lora_state(model),out/'selected_adapter/adapter.pt');atomic_json(out/'selected_adapter/adapter_config.json',json.loads((out/'adapter_config.json').read_text()));proc.save_pretrained(out/'processor')
    atomic_json(out/'training_complete.json',{'selected_checkpoint':'selected_adapter/adapter.pt','selected_val_mil_loss':best,'adapter_sha256':sha256_file(out/'selected_adapter/adapter.pt'),'source_only':True});return out

class TunedDenseScorer:
    def __init__(self,model_path,adapter,device='cuda:0'):
        from transformers import AutoModelForImageTextToText,AutoProcessor
        self.torch=torch;self.device=device;self.model='Qwen/Qwen3.8-27B';self.revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0';self.endpoint='local_transformers'
        self.adapter=Path(adapter);self.cache_root=self.adapter/'dense_inference_cache';self.cache_root.mkdir(exist_ok=True);self.processor=AutoProcessor.from_pretrained(self.adapter.parent/'processor');tok=self.processor.tokenizer;self.id0=tok.encode('0',add_special_tokens=False)[0];self.id1=tok.encode('1',add_special_tokens=False)[0]
        base=AutoModelForImageTextToText.from_pretrained(model_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device);cfg=json.loads((self.adapter/'adapter_config.json').read_text());inject_native_lora(base,cfg['rank'],cfg['alpha'],cfg['targets']);base.load_state_dict(torch.load(self.adapter/'adapter.pt',map_location='cpu'),strict=False);self.net=base.eval()
    def score(self,paths,indices):
        key=hashlib.sha256(('_'.join(sha256_file(p) for p in paths)+sha256_file(self.adapter/'adapter.pt')).encode()).hexdigest();f=self.cache_root/(key+'.json')
        if f.exists():x=json.loads(f.read_text());x['cache_hit']=True;return x
        try:
            with self.torch.no_grad(): z=_binary_logit(self.net,_inputs(self.processor,paths),self.id0,self.id1,self.device);s=float(self.torch.sigmoid(z).cpu())
            x={'success':True,'score':s,'logit':float(z.cpu()),'forced_choice':'1' if s>=.5 else '0','cache_hit':False}
        except Exception as e:x={'success':False,'error':repr(e),'cache_hit':False}
        atomic_json(f,x);return x
