#!/bin/sh
# fill this in with your OpenMM conda environment
source deactivate
source activate py37

#export OPENMM_CUDA_COMPILER=$(which nvcc)
#export CUDA_CACHE_PATH=${TEMPDIR}

python run_openMM.py > test_Tu_1128.log
#python run_openMM.py > test_Tu_0922.log
#python run_openMM.py start4_400tmpatfsi_10A_3.pdb --volt 0 --nsec 10  --type MC_equil > mc3_eq_tmpa_tfsi_10A_10ns.log
#python run_openMM.py start4_400tmpatfsi_10A.pdb --volt 0 --nsec 10  --type MC_equil > mc_eq_tmpa_tfsi_10A_10ns.log
#python run_openMM.py start4_400tmpatfsi_10A.pdb --volt 0 --nsec 10  --type MC_equil > mc4_eq_tmps_tfsi_flat_10ns.log
#python run_openMM.py start_neat_tmpa_tfsi_flat.pdb --volt 0 --nsec 10  --type MC_equil > mc_eq_tmps_tfsi_flat_10ns.log
#python run_openMM.py start4_neat_tmpa_tfsi_10A.pdb --volt 0 --nsec 1  --type MC_equil > mc_eq_tmps_tfsi_10A_1ns.log
#python run_openMM.py npt_eq_400otf_flat.pdb --volt 0 --nsec 0.1  --type MC_equil > mc_eq_bmiotf_100ps.log
#python run_openMM_mc.py start4_400tmpatfsi_nogrph_10A.pdb --detail mc_eq_400tmpatfsi_nogrph_8A > mc_eq_tmpa_tfsi_nogrph_8A_10ns.log
