#!/bin/bash
source activate cuda

export CUDA_VISIBLE_DEVICES=0

python run_openMM.py > energy.log
