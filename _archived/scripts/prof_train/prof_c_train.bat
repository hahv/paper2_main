@REM python main.py --model efficientvit_b1.r224_in1k --batch-size 384 --epochs 200 --data-path E:\watcam_mini_split --dist-eval --drop-path 0.3 --train-interpolation bilinear --num_workers 10 --output_dir ./out --min_acc_save_ckpt 5.0

@REM python main.py --model twins_svt_small --batch-size 768 --data-path C:/datasetfc/dfire_4++_c --dist-eval --drop-path 0.3 

@REM ./distributed_train.sh 4 /data/imagenet --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 --amp -j 4
@REM --resume %RESUME% ^
@echo off

set MODEL=hgnetv2_b5.ssld_stage2_ft_in1k
::set MODEL=twins_svt_small
set BATCH_SIZE=64
set SCHED=cosine
set EPOCHS=100
set WARMUP_EPOCHS=5
set DATA_PATH=C:/datasets/ufire/20250806
set DROP_PATH=0.3
set LR=0.0001
set REMODE=pixel
set RESUME=C:/Users/user/Desktop/workC/timm/pretrained/hgnetv2_b5.ssld_stage2_ft_in1k-20250601-96-98.80.pth.tar

python c_train.py ^
    --model %MODEL% ^
    --sched %SCHED% ^
    --batch-size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --warmup-epochs %WARMUP_EPOCHS% ^
    --data-dir %DATA_PATH% ^
    --drop-path %DROP_PATH% ^
    --input-size 3 360 640 ^
    --crop-pct 1.00 ^
    --lr %LR% ^
    --pretrained True ^
    --workers 4 ^
    --log-interval 400 ^
    --grad-accum-steps 2 ^
    --cutmix 0 ^
    --mixup 0 ^
    --amp

::    --resume %RESUME% ^
::    --cutmix 0 ^
::   --input-size 3 324 576 ^
::   --crop-pct 0.90 ^
