#!/bin/bash


# . $GDVB/scripts/init_conda.sh
# . $SwarmHost/.env.d/openenv.sh
# conda activate swarmhost

# python -m swarm_host $@

# conda deactivate
conda run -n swarmhost python -m swarm_host $@
