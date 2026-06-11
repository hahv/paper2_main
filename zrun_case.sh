
# ! MUST DO: config/zruns/select_opt_methods.yaml
# !- choose the METHODs to RUN or to OPTIMIZE
## ! PRECOMPUTED:
#  ! TEST set: zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorTest__mt_no_temp_method__af4b0d32a3d2__20260330.205208
#  ! VAL set: zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260421.140547

# !======================= Sanity Check ==========================

#  ! run selected methods (sanity check with ufire02 small dataset)
# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_02.yaml" \
#     -pc "zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorTest__mt_no_temp_method__af4b0d32a3d2__20260330.205208"

# !======================= OTPIM - Valid set ======================
# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_val.yaml" \
#     -pc "zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260421.140547"


# on test set
# run eager in val set for selected configs
# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_val.yaml" \
#     -pc "zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260421.140547"

# ./zrun.sh -p val -opt

# !====================== TEST SET ======================

# ! RUN:  AccMotionDet (Eager Mode) on test set for optimized config
# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_test.yaml" \
#     -pc "zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorTest__mt_no_temp_method__af4b0d32a3d2__20260330.205208"


# ./zrun.sh -p test -opt

# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_haze.yaml" \
#     -pc "zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorTest__mt_no_temp_method__af4b0d32a3d2__20260330.205208"

# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_haze.yaml" \
#     -pc "zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorTest__mt_no_temp_method__af4b0d32a3d2__20260330.205208"

# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_haze.yaml"

# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_full.yaml"

# python ./zbin/run_multi.py \
#     --base_yaml "config/zruns/run_base.yaml" \
#     --sweep_yaml "config/zruns/select_ds_ufire_eager.yaml"

python ./zbin/run_multi.py \
    --base_yaml "config/zruns/run_base.yaml" \
    --sweep_yaml "config/zruns/select_ds_ufire_quality.yaml"