#!/usr/bin/env bash

python parameter_sweep_regression.py -c ./config/weight_pruning_prior_1/wine-quality-red.yaml           -ds wine-quality-red             -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &
python parameter_sweep_regression.py -c ./config/weight_pruning_prior_1/bostonHousing.yaml              -ds bostonHousing                -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &
python parameter_sweep_regression.py -c ./config/weight_pruning_prior_1/yacht.yaml                      -ds yacht                        -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &
python parameter_sweep_regression.py -c ./config/weight_pruning_prior_1/protein-tertiary-structure.yaml -ds protein-tertiary-structure   -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &

wait
echo "All Finished"