#!/usr/bin/env python3
import base64,json,re,time
from pathlib import Path
import httpx
MODEL="Qwen/Qwen3.8-27B"; URL="http://127.0.0.1:8000/v1/chat/completions"
SCHEMA={"type":"json_schema","json_schema":{"name":"clip_anomaly_assessment","strict":True,"schema":{"type":"object","additionalProperties":False,"required":["sampled_frame_scores","clip_score","confidence","decision","visible_evidence","reason"],"properties":{"sampled_frame_scores":{"type":"array","minItems":8,"maxItems":8,"items":{"type":"integer","minimum":0,"maximum":100}},"clip_score":{"type":"integer","minimum":0,"maximum":100},"confidence":{"type":"number","minimum":0,"maximum":1},"decision":{"type":"string","enum":["normal","uncertain","abnormal"]},"visible_evidence":{"type":"array","items":{"type":"string"}},"reason":{"type":"string"}}}}}
PROMPT="Analyze the following 8 surveillance frames strictly in chronological order. Each image is labeled with its original frame index. Score visible anomaly evidence only; do not infer or invent unseen behavior. Return only the requested JSON object, without Markdown or surrounding text."
def natural(p):return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)',p.name)]
def sample_frames(frame_dir):
 files=sorted([p for p in Path(frame_dir).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}],key=natural)
 if len(files)<8:raise ValueError(f'need >=8 frames: {frame_dir}')
 idx=[round(i*(len(files)-1)/7) for i in range(8)];return [(i,files[i]) for i in idx]
def data_url(path):return 'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()
def payload(samples,repair=False):
 content=[{"type":"text","text":PROMPT+(" Previous formatting was invalid; emit exactly one valid JSON object." if repair else "")}]
 for index,path in samples:content.extend([{"type":"text","text":f"Original frame index: {index+1}; filename: {path.name}"},{"type":"image_url","image_url":{"url":data_url(path)}}])
 return {"model":MODEL,"messages":[{"role":"user","content":content}],"temperature":0,"max_tokens":512,"stream":True,"response_format":SCHEMA,"chat_template_kwargs":{"enable_thinking":False}}
def validate(x):
 assert set(x)=={"sampled_frame_scores","clip_score","confidence","decision","visible_evidence","reason"};assert len(x['sampled_frame_scores'])==8;assert all(type(v) is int and 0<=v<=100 for v in x['sampled_frame_scores']);assert type(x['clip_score']) is int and 0<=x['clip_score']<=100;assert 0<=x['confidence']<=1;assert x['decision'] in {'normal','uncertain','abnormal'};assert isinstance(x['visible_evidence'],list) and isinstance(x['reason'],str)
def request(samples,repair=False,timeout=600):
 started=time.perf_counter();first=None;parts=[]
 with httpx.stream('POST',URL,headers={'Authorization':'Bearer EMPTY'},json=payload(samples,repair),timeout=timeout) as response:
  response.raise_for_status()
  for line in response.iter_lines():
   if not line.startswith('data: ') or line=='data: [DONE]':continue
   obj=json.loads(line[6:]);delta=obj['choices'][0].get('delta',{}).get('content')
   if delta:
    if first is None:first=time.perf_counter()
    parts.append(delta)
 raw=''.join(parts);end=time.perf_counter()
 try:parsed=json.loads(raw);validate(parsed);ok=True
 except Exception:
  match=re.search(r'\{.*\}',raw,re.S);parsed=json.loads(match.group(0)) if match else None
  try:validate(parsed);ok=True
  except Exception:ok=False
 return {'raw':raw,'parsed':parsed,'valid':ok,'ttft_seconds':None if first is None else first-started,'total_seconds':end-started,'repair':repair}
def request_with_retry(samples):
 first=request(samples)
 if first['valid']:return {'attempts':[first],'result':first}
 second=request(samples,True);return {'attempts':[first,second],'result':second}
