#!/bin/bash


# . $GDVB/scripts/init_conda.sh
# . $R4V/.env.d/openenv.sh

# python -m r4v $@
# /home/nguyenho/miniconda3/envs/r4v/bin/python -m r4v $@

# python -m r4v $@
conda run -n adagdvb python -m r4v $@

