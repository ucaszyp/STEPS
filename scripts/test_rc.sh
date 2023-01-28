#!/bin/bash
# for j in $(seq 1 1 3)
# do
# for i in $(seq 28 1 28)
# do
# echo $i
CUDA_VISIBLE_DEVICES=4 python test_robotcar_disp.py night rnw_rc best/checkpoint_epoch=28.ckpt --test 1
cd evaluation
python eval_robotcar.py night
# cd ..
# # done
# done