#!/bin/bash

# AdaGDVB
conda env create --name adagdvb -f .env.d/env.yml
. .env.d/openenv.sh

# GDVB
git clone https://github.com/edwardxu0/GDVB.git $AdaGDVB/lib/GDVB
conda env create --name gdvb -f $GDVB/.env.d/env.yml

# R4V Run r4v with AdaGDVB env
# git clone https://github.com/edwardxu0/R4V.git $GDVB/lib/R4V
# conda env create --name r4v -f $R4V/.env.d/env.yml

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

# mn-bab
git clone --recursive https://github.com/eth-sri/mn-bab.git $SwarmHost/lib/mnbab
conda env create --name mnbab -f $SwarmHost/envs/mnbab.yml
