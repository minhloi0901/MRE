#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
eval "$(conda shell.bash hook)"
conda activate dire
EXP_NAME="imagenet"
DATASETS="imagenet"
DATASETS_TEST="imagenet"
python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST
