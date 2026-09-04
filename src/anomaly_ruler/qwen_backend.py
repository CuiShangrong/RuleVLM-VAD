from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .core import atomic_json, stable_hash, sha256_file


class QwenBackend:
    def __init__(self, endpoint, model, revision, cache_root, *, timeout=300, max_tokens=512):
        self.endpoint=endpoint; self.model=model; self.revision=revision
        self.cache_root=Path(cache_root); self.cache_root.mkdir(parents=True, exist_ok=True)
        self.timeout=timeout; self.max_tokens=max_tokens

    def _call(self, messages, key_payload, kind, max_tokens=None):
        effective_tokens=self.max_tokens if max_tokens is None else int(max_tokens)
        key_data={"model":self.model,"revision":self.revision,"kind":kind,"payload":key_payload}
        # Preserve existing visual-description cache keys; overridden official generation
        # limits must be isolated from prior truncated responses.
        if max_tokens is not None: key_data["max_tokens"]=effective_tokens
        key=stable_hash(key_data)
        path=self.cache_root/kind/f"{key}.json"
        if path.exists():
            saved=json.loads(path.read_text()); saved["cache_hit"]=True; return saved
        payload={"model":self.model,"messages":messages,"temperature":0,"max_tokens":effective_tokens,
                 "chat_template_kwargs":{"enable_thinking":False}}
        request=urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
            headers={"Content-Type":"application/json","Authorization":"Bearer EMPTY"})
        start=time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body=json.loads(response.read())
            text=body["choices"][0]["message"]["content"].strip()
            result={"success":True,"text":text,"elapsed_seconds":time.monotonic()-start,
                    "usage":body.get("usage",{}),"cache_hit":False,"key":key}
        except Exception as exc:
            result={"success":False,"error":repr(exc),"elapsed_seconds":time.monotonic()-start,
                    "cache_hit":False,"key":key}
        atomic_json(path,result)
        return result

    @staticmethod
    def _image_url(path):
        suffix=Path(path).suffix.lower(); mime="image/png" if suffix==".png" else "image/jpeg"
        return f"data:{mime};base64,"+base64.b64encode(Path(path).read_bytes()).decode()

    def describe_image(self, image_path):
        # Exact official CogVLM visual-perception query, mapped to Qwen.
        query="How many people are in the image and what is each of them doing? What are in the images other than people? Think step by step"
        content=[{"type":"text","text":query},{"type":"image_url","image_url":{"url":self._image_url(image_path)}}]
        return self._call([{"role":"user","content":content}], {"image_sha256":sha256_file(image_path),"query":query}, "description")

    def chat(self, messages, kind, key_payload, max_tokens=None):
        return self._call(messages,key_payload,kind,max_tokens=max_tokens)
