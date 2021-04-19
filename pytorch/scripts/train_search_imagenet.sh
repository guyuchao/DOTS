#!/bin/bash

source /home/guyuchao/anaconda3/bin/activate ofaseg

cd ..

GPU_IDS=0,1,2,3
CUDA_VISIBLE_DEVICES=${GPU_IDS} OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 train_search_imagenet.py
