#!/usr/bin/env bash

#python parameter_sweep.py -c ./config/weight_pruning_hyperprior/wine-quality-red.yaml           -ds wine-quality-red             -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &
#python parameter_sweep.py -c ./config/weight_pruning_hyperprior/bostonHousing.yaml              -ds bostonHousing                -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &
#python parameter_sweep.py -c ./config/weight_pruning_hyperprior/yacht.yaml                      -ds yacht                        -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &

wait

#python parameter_sweep.py -c ./config/weight_pruning_hyperprior/concrete.yaml                   -ds concrete                     -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/weight_pruning_hyperprior/protein-tertiary-structure.yaml -ds protein-tertiary-structure   -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &

wait

#python parameter_sweep.py -c ./config/weight_pruning_hyperprior/energy.yaml                     -ds energy                       -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &
#python parameter_sweep.py -c ./config/weight_pruning_hyperprior/kin8nm.yaml                     -ds kin8nm                       -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &
#python parameter_sweep.py -c ./config/weight_pruning_hyperprior/naval-propulsion-plant.yaml     -ds naval-propulsion-plant       -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &

wait

#python parameter_sweep.py -c ./config/weight_pruning_hyperprior/power-plant.yaml                -ds power-plant                  -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &
#python parameter_sweep.py -c ./config/weight_pruning_hyperprior/mnist.yaml                      -ds mnist                        -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/presplit/ -cm weight_pruning_hyperprior3 -nd --gpu 4 5 6 7 &

wait
echo "All Finished"