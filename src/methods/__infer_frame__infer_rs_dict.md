
# Infer RS Dictionary

```python

"infer_rs" :{

"mt_cfg":
{
│   'block_active_thresh': 0.1,
│   'block_size': 16,
│   'motion': {'name': 'framediff_det.FrameDiffDet', 'params': {'diff_thresh': 30}},
│   'rule_params': {
│   │   'active_ratio_threshold': 0.2,
│   │   'wavelet_channel': 'r',
│   │   'wavelet_energy_thres': 50.0
│   },
│   'scale_factor': 1.0
},

"mt_proc":

{
│   'vis_frame': array([[[37, 38, 33],
        ....
│   │   [ 0,  0,  0]]], shape=(1088, 1920, 3), dtype=uint8),
│   'motion_mask_frame': array([[0, 0, 0, ..., 0, 0, 0],
│      [0, 0, 0, ..., 0, 0, 0],
│      ...,
│      [0, 0, 0, ..., 0, 0, 0]], shape=(1088, 1920), dtype=uint8),
│   'block_info': [
│   │   {
│   │   │   'block_id': (4, 24),
│   │   │   'rule_dict': {
│   │   │   │   'FireOrSmokeCheck -> SmokeCheck': RuleResult(
│   │   │   │   │   rule_name='SmokeCheck',
│   │   │   │   │   status=<RuleStatus.PASS: 'PASS'>,
│   │   │   │   │   details={
│   │   │   │   │   │   'percent_smoke': 0.453125,
│   │   │   │   │   │   'threshold': 0.1,
│   │   │   │   │   │   'msg': 'Smoke HSV Ratio 0.45 > 0.1'
│   │   │   │   │   },
│   │   │   │   │   sub_results=[]
│   │   │   │   )
│   │   │   }
│   │   },
│   │   {
│   │   │   'block_id': (5, 24),
│   │   │   'rule_dict': {
│   │   │   │   'FireOrSmokeCheck -> SmokeCheck': RuleResult(
│   │   │   │   │   rule_name='SmokeCheck',
│   │   │   │   │   status=<RuleStatus.PASS: 'PASS'>,
│   │   │   │   │   details={
│   │   │   │   │   │   'percent_smoke': 0.2578125,
│   │   │   │   │   │   'threshold': 0.1,
│   │   │   │   │   │   'msg': 'Smoke HSV Ratio 0.26 > 0.1'
│   │   │   │   │   },
│   │   │   │   │   sub_results=[]
│   │   │   │   )
│   │   │   }
│   │   },
│   │   {
│   │   │   'block_id': (6, 24),
│   │   │   'rule_dict': {
│   │   │   │   'FireOrSmokeCheck -> SmokeCheck': RuleResult(
│   │   │   │   │   rule_name='SmokeCheck',
│   │   │   │   │   status=<RuleStatus.PASS: 'PASS'>,
│   │   │   │   │   details={
│   │   │   │   │   │   'percent_smoke': 0.47265625,
│   │   │   │   │   │   'threshold': 0.1,
│   │   │   │   │   │   'msg': 'Smoke HSV Ratio 0.47 > 0.1'
│   │   │   │   │   },
│   │   │   │   │   sub_results=[]
│   │   │   │   )
│   │   │   }
│   │   },
│   │   {
│   │   │   'block_id': (7, 24),
│   │   │   'rule_dict': {
│   │   │   │   'FireOrSmokeCheck -> SmokeCheck': RuleResult(
│   │   │   │   │   rule_name='SmokeCheck',
│   │   │   │   │   status=<RuleStatus.PASS: 'PASS'>,
│   │   │   │   │   details={
│   │   │   │   │   │   'percent_smoke': 0.7578125,
│   │   │   │   │   │   'threshold': 0.1,
│   │   │   │   │   │   'msg': 'Smoke HSV Ratio 0.76 > 0.1'
│   │   │   │   │   },
│   │   │   │   │   sub_results=[]
│   │   │   │   )
│   │   │   }
│   │   },
│   │   {
│   │   │   'block_id': (8, 24),
│   │   │   'rule_dict': {
│   │   │   │   'FireOrSmokeCheck -> SmokeCheck': RuleResult(
│   │   │   │   │   rule_name='SmokeCheck',
│   │   │   │   │   status=<RuleStatus.PASS: 'PASS'>,
│   │   │   │   │   details={
│   │   │   │   │   │   'percent_smoke': 0.328125,
│   │   │   │   │   │   'threshold': 0.1,
│   │   │   │   │   │   'msg': 'Smoke HSV Ratio 0.33 > 0.1'
│   │   │   │   │   },
│   │   │   │   │   sub_results=[]
│   │   │   │   )
│   │   │   }
│   │   }
│   ]
        }
}
```
