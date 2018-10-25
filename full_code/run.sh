#!/usr/bin/env bash
PADDLE_PATH=/home/yi/code/paddle/idl/paddle/paddle
export PATH=$PADDLE_PATH:$PATH
export PYTHONPATH=$PADDLE_PATH/output/pylib:$PYTHONPATH
export PYTHONPATH=./paddle/data_provider:$PYTHONPATH

TASK=coco_cap
DATASET=data_coco_cap
#TASK=coco_vqa
#DATASET=data_coco_vqa
#TASK=coco_cap_vqa
#DATASET=data_coco_cap\ data_coco_vqa
#TASK=coco_qa_cn
#TASK=coco_cap_cn
#TASK=tieba_dia_cn
#TASK=coco_qa_cap_noword
#TASK=coco_qa_cap_cn
#TASK=coco_qa_cap_dia_cn
#TASK=imagenet_cap
#TASK=imagenet_qa
#TASK=imagenet_qa_cap
#TASK=imagenet_qa_cap_noword

#TRAIN_CONF=./paddle/conf/image_qa.py
#TEST_CONF=./paddle/conf/image_qa_gen.py
#TRAIN_CONF=./paddle/conf/chat.py
#TEST_CONF=./paddle/conf/chat_gen.py
#TRAIN_CONF=./paddle/conf/lstm_v2.py
#TEST_CONF=./paddle/conf/lstm_v2_gen.py
#TRAIN_CONF=./paddle/conf/lstm_recurrentunit.py
#TEST_CONF=./paddle/conf/lstm_recurrentunit_gen.py
#TRAIN_CONF=./paddle/conf/lstm_yi.py
#TEST_CONF=./paddle/conf/lstm_yi_gen.py
#TRAIN_CONF=./paddle/conf/lstm_yi_recurrentunit.py
#TEST_CONF=./paddle/conf/lstm_yi_recurrentunit_gen.py
#TRAIN_CONF=./paddle/conf/lstm_v1.py
#TEST_CONF=./paddle/conf/lstm_v1_gen.py
#TRAIN_CONF=./paddle/conf/lstm_v2_junhua.py
#TEST_CONF=./paddle/conf/lstm_v2_junhua_gen.py
#TRAIN_CONF=./paddle/conf/lstm_v2.py
#TEST_CONF=./paddle/conf/lstm_v2_gen.py
TRAIN_CONF=./paddle/conf/image_qa.py
TEST_CONF=./paddle/conf/image_qa_gen.py
#TRAIN_CONF=./paddle/conf/image_qa_attention.py
#TEST_CONF=./paddle/conf/image_qa_attention.py
#TRAIN_CONF=./paddle/conf/image_qa_gate.py
#TEST_CONF=./paddle/conf/image_qa_gate_gen.py
#TRAIN_CONF=./paddle/conf/image_qa_gate2.py
#TEST_CONF=./paddle/conf/image_qa_gate2_gen.py
#TRAIN_CONF=./paddle/conf/image_qa_regularize_more.py
#TEST_CONF=./paddle/conf/image_qa_gen.py
#TRAIN_CONF=./paddle/conf/caption_qa.py
#TEST_CONF=./paddle/conf/caption_qa_gen.py


#FEAT_TYPE=vgg_fc7_4096
#FEAT_TYPE=vgg_fc7_4096_oversample
#FEAT_TYPE=google_pool5_1024
#FEAT_TYPE=google_pool5_1024_oversample
#FEAT_TYPE=google_3072
#FEAT_TYPE=resnet_pool5_2048_oversample
FEAT_TYPE=resnet152_pool5_2048_oversample
#FEAT_TYPE=resnet152_res5c_spatial_2048
FEAT_DIM=2048

python visual_language.py --task $TASK --dataset $DATASET --train_conf $TRAIN_CONF --test_conf $TEST_CONF --image_feat_type $FEAT_TYPE --image_feat_dim $FEAT_DIM --train_epoch 3 --use_gpu --trainer_count 2 --debug
#TRAIN_EPOCH=31
#python visual_language.py --task $TASK --dataset $DATASET --train_conf $TRAIN_CONF --test_conf $TEST_CONF --image_feat_type $FEAT_TYPE --image_feat_dim $FEAT_DIM --train_epoch $TRAIN_EPOCH --use_gpu --trainer_count 1 --email yangyi05@baidu.com 
