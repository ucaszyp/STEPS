# WORKDIR=checkpoints

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 train.py \
--config rnw_ns \
--gpus 8 \