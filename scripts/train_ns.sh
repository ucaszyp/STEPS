# WORKDIR=checkpoints
# mkdir $WORKDIR
# cp -r configs/rnw_ns.yaml  $WORKDIR
# cp -r models/rnw.py $WORKDIR
# cp -r train.sh $WORKDIR
# cp -r train.py $WORKDIR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 train.py \
--config rnw_ns \
--gpus 8 \
# --work_dir $WORKDIR