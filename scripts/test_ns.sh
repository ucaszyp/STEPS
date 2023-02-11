#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python test_nuscenes_disp.py night steps_ns best/ns_best_wo_denoise.ckpt --test 1
cd evaluation
python eval_nuscenes.py night
cd ..
# done