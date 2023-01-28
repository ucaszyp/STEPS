WORKDIR=./exp-rc/night6
mkdir $WORKDIR
cp -r configs/rnw_rc.yaml  $WORKDIR
cp -r models/rnw.py $WORKDIR
cp -r train.sh $WORKDIR
cp -r train.py $WORKDIR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
python3 train.py \
--config rnw_rc \
--gpus 7 \
--work_dir $WORKDIR