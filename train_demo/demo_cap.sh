#!/usr/bin/env bash
PADDLE_PATH=/home/yi/code/paddle/idl/paddle/paddle
export PATH=$PADDLE_PATH:$PATH
export PYTHONPATH=$PADDLE_PATH/output/pylib:$PYTHONPATH
export PYTHONPATH=./paddle/data_provider:$PYTHONPATH

# Demo on training and testing image captioning
python visual_language.py --cache_dir cache --task demo_cap --dataset data_demo_cap --train_conf ./paddle/conf/train_conf.py --test_conf ./paddle/conf/test_conf.py --image_feat_type resnet152_pool5_2048_oversample --image_feat_dim 2048 --train_epoch 81 --use_gpu --trainer_count 2 
