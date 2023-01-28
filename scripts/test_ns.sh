#!/usr/bin/env bash
# for i in $(seq 15 1 15)
# do
CUDA_VISIBLE_DEVICES=7 python test_nuscenes_disp.py night rnw_ns best/ns_denoise_best.ckpt --test 1
cd evaluation
python eval_nuscenes.py night
cd ..
# done