#!/bin/bash

# . ${SwarmHost}/scripts/init_conda.sh

conda run -n marabou python $SwarmHost/lib/marabou/resources/runMarabou.py $@
