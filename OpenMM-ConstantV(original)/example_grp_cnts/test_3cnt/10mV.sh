#!/bin/sh

# fill this in with your OpenMM conda environment
source deactivate
source activate py37
#conda activate py37
#source activate py37

#export CUDA_VISIBLE_DEVICES=1

python run_openMM.py npt_eq_neat_bmitfsi_3cnt.pdb --volt 0.01 --nsec 100  --type Constant_V > md_bmim_tfsi_10mV_100ns.log
