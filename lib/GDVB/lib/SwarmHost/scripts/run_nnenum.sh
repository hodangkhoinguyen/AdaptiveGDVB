#!/bin/bash

# . ${SwarmHost}/scripts/init_conda.sh
# conda activate nnenum

# export OPENBLAS_NUM_THREADS=1 
# export OMP_NUM_THREADS=1

# export PROJECT_ROOT=`pwd`
# cd $SwarmHost/lib/nnenum/src/
# cmd="python -m nnenum.nnenum $@"
# echo $cmd
# $cmd
# cd $PROJECT_ROOT

# conda deactivate
cd $SwarmHost/lib/nnenum/src/
conda run -n nnenum python -m nnenum.nnenum $@
cd $PROJECT_ROOT