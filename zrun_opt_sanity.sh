#!/bin/bash
source ./zutils.sh
set_optim_outdir "zout/zoptim_test"

./zrun.sh \
    -r config/zruns/zrun_cfg_opt_test.yaml \
    -opt \
    -pre ./zout/zoptim_test/MainPC__ds_UFireIndoor2__mt_no_temp_method__af4b0d32a3d2__20260316.172122