#!/usr/bin/env bash

python parameter_sweep_regression.py -c ./config/weight_pruning_hyperprior/wine-quality-red.yaml           -ds wine-quality-red             -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning_hyperprior -nd --gpu 0 1 2 3 &
python parameter_sweep_regression.py -c ./config/weight_pruning_hyperprior/bostonHousing.yaml              -ds bostonHousing                -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning_hyperprior -nd --gpu 0 1 2 3 &
python parameter_sweep_regression.py -c ./config/weight_pruning_hyperprior/yacht.yaml                      -ds yacht                        -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning_hyperprior -nd --gpu 0 1 2 3 &
python parameter_sweep_regression.py -c ./config/weight_pruning_hyperprior/protein-tertiary-structure.yaml -ds protein-tertiary-structure   -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning_hyperprior -nd --gpu 0 1 2 3 &

wait
echo "All Finished"