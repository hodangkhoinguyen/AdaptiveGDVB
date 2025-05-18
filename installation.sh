#!/bin/bash

# AdaGDVB
conda env create --name adagdvb -f .env.d/env.yml
. .env.d/openenv.sh

# SwarmHost
git clone https://github.com/edwardxu0/SwarmHost.git $GDVB/lib/SwarmHost
wget https://raw.githubusercontent.com/dlshriver/dnnv/main/tools/resmonitor.py $SwarmHost/lib
conda env create --name swarmhost -f $SwarmHost/.env.d/env.yml

# alpha-beta-crown
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git $SwarmHost/lib/abcrown
conda env create --name abcrown -f $SwarmHost/lib/abcrown/complete_verifier/environment.yaml

# neuralsat
git clone https://github.com/dynaroars/neuralsat.git $SwarmHost/lib/neuralsat
conda env create --name neuralsat -f $SwarmHost/lib/neuralsat/env.yaml

# nnenum
git clone https://github.com/stanleybak/nnenum.git $SwarmHost/lib/nnenum
conda env create --name nnenum -f $SwarmHost/envs/nnenum.yml

# marabou
git clone https://github.com/NeuralNetworkVerification/Marabou.git $SwarmHost/lib/marabou
conda create --name marabou python==3.9
cd $SwarmHost/lib/marabou
