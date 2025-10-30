#!/bin/bash
source activate py37

export CUDA_VISIBLE_DEVICES=0

python run_openMM.py > energy.log
