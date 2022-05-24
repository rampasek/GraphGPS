#!/usr/bin/env bash

trap "kill 0" SIGINT

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $1 == verbose ]]; then
    echo "$(tput setaf 2)Verbose set, dumping log to stdout$(tput sgr 0)"
    redirect=''
else
    echo "$(tput setaf 2)Verbose unset, output muted." \
         "Specify 'verbose' as the first argument to show log.$(tput sgr 0)"
    redirect='> /dev/null'
fi

if [[ $CONDA_DEFAULT_ENV != graphgps ]]; then
    source ~/.bashrc
    conda activate graphgps
fi

echo homdir=${homedir}


function test_dataset {
    trap "kill 0" ERR

    dataset_format=$1
    task_type=$2

    if [[ ! -z $3 ]]; then
        namestr="dataset.name ${3}"
    fi

    echo "Start testing ${dataset_format}, task_type=${task_type}"

    main_prog="python main.py --cfg tests/configs/node/default.yaml --repeat 1"
    eval "${main_prog} dataset.format ${dataset_format} ${namestr} " \
         "dataset.split_mode standard dataset.task_type ${task_type} " \
         "optim.max_epoch 5 " \
         "out_dir tests/results/${dataset_format} ${redirect}"
}

test_dataset PyG-Planetoid classification CiteSeer
test_dataset PyG-Planetoid classification Cora
test_dataset PyG-Planetoid classification PubMed

#test_dataset PyG-WikipediaNetwork classification chameleon
#test_dataset PyG-WikipediaNetwork classification squirrel

echo ALL PASSED!
