#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != graphgps ]]; then
    source ~/.bashrc
    conda activate graphgps
fi

echo "homdir=${homedir}"

python main.py --cfg tests/configs/node/citeseer.yaml --repeat 3
python main.py --cfg tests/configs/node/cora.yaml --repeat 3
python main.py --cfg tests/configs/node/pubmed.yaml --repeat 3

python main.py --cfg tests/configs/node/ogbn-arxiv.yaml --repeat 1
