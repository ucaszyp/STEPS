#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python test_robotcar_disp.py night steps_rc best/rc_best.ckpt --test 1
cd evaluation
python eval_robotcar.py night

# done