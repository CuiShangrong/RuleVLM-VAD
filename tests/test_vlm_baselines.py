import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'UR-DMU'))
from vlm_baselines.common import uniform_indices,uniform_indices_fixed_count,windows,json_score,DIRECT_PROMPT
from vlm_baselines.dense_zeroshot import temporal_centers,context_bounds,interpolate_centers

def test_uniform_sampling():
    assert uniform_indices(0,32,4)==[0,10,21,31]
    assert uniform_indices(5,7,4)==[5,6]
def test_fixed_eight_sampling_including_short_tail():
    assert uniform_indices_fixed_count(0,4,8)==[0,0,1,1,2,2,3,3]
    assert uniform_indices_fixed_count(240,241,8)==[240]*8
    assert uniform_indices_fixed_count(0,240,8)==[0,34,68,102,137,171,205,239]
def test_window_cover():
    assert windows(65,32,32)==[(0,32),(32,64),(64,65)]
def test_continuous_json_score():
    score,obj=json_score('{"anomaly_score": 37, "confidence":0.8,"decision":"uncertain","visible_evidence":[]}')
    assert score==.37 and obj['decision']=='uncertain'
def test_direct_has_no_gt_access():
    source=(Path(__file__).resolve().parents[1]/'vlm_baselines/common.py').read_text()
    assert 'test_frame_mask' not in source and 'roc_auc_score' not in source and 'average_precision_score' not in source
def test_zeroshot_is_target_only_and_has_fixed_tail_policy():
    source=(Path(__file__).resolve().parents[1]/'vlm_baselines/common.py').read_text()
    assert 'source_data_used' in source and "'source_data_used': False" in source
    assert 'retain_and_repeat_to_8' in source
def test_dense_centers_bounds_and_interpolation():
    assert temporal_centers(33,16)==[0,16,32]
    assert context_bounds(0,100,240)==(0,100)
    assert context_bounds(120,500,240)==(0,240)
    assert np.allclose(interpolate_centers([0,16,32],[0.,.5,1.],33)[[0,8,16,24,32]],[0,.25,.5,.75,1])
