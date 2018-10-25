#!/usr/bin/env bash
PADDLE_PATH=/home/yi/code/paddle/idl/paddle/paddle
export PATH=$PADDLE_PATH:$PATH
export PYTHONPATH=$PADDLE_PATH/output/pylib:$PYTHONPATH

paddle_trainer --config=conf.py --use_gpu=1 --trainer_count=2 --test_period=0 --num_passes=44 --save_dir=model --job=test --test_pass=43 --config_args=file_list=file.list,result_file=result.txt,dict_file=dict.txt,dict_pkl=dict.pkl,img_feat_dim=4096,batch_size=2
