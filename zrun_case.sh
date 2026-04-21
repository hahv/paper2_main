# ! MUST DO: config/zruns/select_opt_methods.yaml
# !- choose the METHODs to RUN or to OPTIMIZE

## ! PRECOMPUTED:
#  ! TEST set: zout/zruns/MainPC__ds_UFireIndoorTest__mt_no_temp_method__af4b0d32a3d2__20260330.205208
#  ! VAL set: zout/zruns/MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260327.162459

# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_02.yaml"

# on test set
# ./zrun.sh -p test -opt --dry
#  run eager in val set for selected configs
# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_val.yaml" \
#     -pc "zout/zruns/MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260327.162459"

# ./zrun.sh -p val -opt

# ! RUN:  AccMotionDet (Eager Mode) on test set for optimized config
python ./zbin/run_multi.py \
    --base_yaml "config/zruns/run_base.yaml" \
    --sweep_yaml "config/zruns/select_ds_ufire_test.yaml" \
    -pc "zout/zruns/MainPC__ds_UFireIndoorTest__mt_no_temp_method__af4b0d32a3d2__20260330.205208"
