#!/bin/sh

# fill this in with your OpenMM conda environment
source deactivate
source activate py37
#conda activate py37
#source activate py37

#export CUDA_VISIBLE_DEVICES=1

python run_openMM.py start_neat_bmitfsi_3cnt.pdb --volt 0 --nsec 10  --type MC_equil > mc_bmim_tfsi_100ns.log
