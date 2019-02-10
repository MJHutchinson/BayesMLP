#!/usr/bin/env bash

python parameter_sweep.py -c ./config/weight_pruning_prior_1/wine-quality-red.yaml           -ds wine-quality-red             -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/weight_pruning_prior_1/bostonHousing.yaml              -ds bostonHousing                -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &

wait

python parameter_sweep.py -c ./config/weight_pruning_prior_1/yacht.yaml                      -ds yacht                        -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/weight_pruning_prior_1/protein-tertiary-structure.yaml -ds protein-tertiary-structure   -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &

wait

python parameter_sweep.py -c ./config/weight_pruning_prior_1/concrete.yaml                   -ds concrete                     -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/weight_pruning_prior_1/energy.yaml                     -ds energy                       -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &

wait

python parameter_sweep.py -c ./config/weight_pruning_prior_1/kin8nm.yaml                     -ds kin8nm                       -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/weight_pruning_prior_1/naval-propulsion-plant.yaml     -ds naval-propulsion-plant       -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &

wait

python parameter_sweep.py -c ./config/weight_pruning_prior_1/power-plant.yaml                -ds power-plant                  -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/weight_pruning_prior_1/mnist.yaml                      -ds mnist                        -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_prior_1 -nd --gpu 4 5 6 7 &


wait
echo "All Finished"