from __future__ import annotations

import json
from pathlib import Path

from .core import natural_key


SETUPS={"01_to_01":("01","01"),"01_to_08":("01","08"),"08_to_08":("08","08"),
        "08_to_01":("08","01"),"mixed_to_mixed":("mixed","mixed")}


class ShanghaiData:
    def __init__(self, workspace):
        self.workspace=Path(workspace)
        self.manifest=json.loads((self.workspace/'splits/shanghaitech_wvad/test_manifest.json').read_text())
        self.by_id={x['video_id']:x for x in self.manifest}
        train=(self.workspace/'splits/shanghaitech_wvad/SH_Train.txt').read_text().splitlines()
        self.train_abnormal=[Path(x.strip()).stem for x in train[:63] if x.strip()]
        self.train_normal=[Path(x.strip()).stem for x in train[63:] if x.strip()]

    def source_normal_ids(self, source):
        return list(self.train_normal) if source=='mixed' else [x for x in self.train_normal if x.split('_')[0]==source]

    def target_manifest(self,target):
        return list(self.manifest) if target=='mixed' else [x for x in self.manifest if x['video_id'].split('_')[0]==target]

    def test_frames(self,video_id):
        a=self.workspace/'datasets/shanghaitech/testing/frames'/video_id
        b=self.workspace/'work/shanghaitech_frames/normal'/video_id
        root=a if a.is_dir() else b
        paths=sorted([*root.glob('*.jpg'),*root.glob('*.jpeg'),*root.glob('*.png')],key=natural_key)
        expected=self.by_id[video_id]['frame_count']
        if len(paths)!=expected: raise ValueError(f'{video_id}: {len(paths)} frames != manifest {expected}')
        return paths

    def source_frames(self,video_id):
        root=self.workspace/'work/shanghaitech_frames/normal'/video_id
        paths=sorted([*root.glob('*.jpg'),*root.glob('*.jpeg'),*root.glob('*.png')],key=natural_key)
        if not paths: raise FileNotFoundError(f'No source frames for {video_id}: {root}')
        return paths

    def weak_train_frames(self, video_id):
        """Frames for the official WVAD training split; labels are video-level only."""
        if video_id in self.train_normal:
            return self.source_frames(video_id)
        if video_id in self.train_abnormal:
            root=self.workspace/'datasets/shanghaitech/testing/frames'/video_id
            paths=sorted([*root.glob('*.jpg'),*root.glob('*.jpeg'),*root.glob('*.png')],key=natural_key)
            if not paths: raise FileNotFoundError(f'No weak-abnormal train frames: {video_id}')
            return paths
        raise KeyError(f'{video_id} absent from SH_Train.txt')
