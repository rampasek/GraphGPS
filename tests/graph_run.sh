#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != graphgps ]]; then
    source ~/.bashrc
    conda activate graphgps
fi

echo "homdir=${homedir}"

# GNNBenchmarkDataset
python main.py --cfg tests/configs/graph/cifar10.yaml
python main.py --cfg tests/configs/graph/mnist.yaml
python main.py --cfg tests/configs/graph/cluster.yaml
python main.py --cfg tests/configs/graph/pattern.yaml

# OGB
python main.py --cfg tests/configs/graph/ogbg-molhiv.yaml

# ZINC
python main.py --cfg tests/configs/graph/zinc.yaml --repeat 1
python main.py --cfg tests/configs/graph/zinc-LapPE.yaml --repeat 1 seed 42

# MalNet
python main.py --cfg tests/configs/graph/malnettiny.yaml run_multiple_splits [3]
python main.py --cfg tests/configs/graph/malnettiny-LapPE.yaml run_multiple_splits [3]

# TUDataset
python main.py --cfg tests/configs/graph/collab.yaml run_multiple_splits [3]
python main.py --cfg tests/configs/graph/dd.yaml run_multiple_splits [3]
python main.py --cfg tests/configs/graph/enzymes.yaml  # run all 10 folds
python main.py --cfg tests/configs/graph/imdb-binary.yaml run_multiple_splits [0,1]
python main.py --cfg tests/configs/graph/imdb-multi.yaml run_multiple_splits [0,1]
python main.py --cfg tests/configs/graph/proteins.yaml run_multiple_splits [0,1]

### Testing runs with multiple random seeds vs. runs with multiple data splits
# Runs the second CV split (indexed from 0) 3 times with different random seeds
python main.py --cfg tests/configs/graph/nci1.yaml --repeat 3 dataset.split_index 1 run_multiple_splits [] seed 42
# Runs 3 CV splits, each with the same random seed
python main.py --cfg tests/configs/graph/nci1.yaml run_multiple_splits [7,8,9] seed 42
