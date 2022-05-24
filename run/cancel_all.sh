#!/usr/bin/env bash

echo "Cancel all slurm jobs of user: ${USER} ?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) scancel -u $USER && break;;
        No ) echo "Aborted" && break;;
    esac
done