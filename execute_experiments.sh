#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6
python experiment_parametrise_regression.py -c ./config/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm christsmas-sweep-1 &
python experiment_parametrise_regression.py -c ./config/bostonHousing.yaml              -ds bostonHousing               -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm christsmas-sweep-1 &
python experiment_parametrise_regression.py -c ./config/concrete.yaml                   -ds concrete                    -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm christsmas-sweep-1 &
export CUDA_VISIBLE_DEVICES=2
python experiment_parametrise_regression.py -c ./config/kin8nm.yaml                     -ds kin8nm                      -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm christsmas-sweep-1 &
python experiment_parametrise_regression.py -c ./config/naval-propulsion-plant.yaml     -ds naval-propulsion-plant      -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm christsmas-sweep-1 &
python experiment_parametrise_regression.py -c ./config/power-plant.yaml                -ds power-plant                 -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm christsmas-sweep-1 &
export CUDA_VISIBLE_DEVICES=3
python experiment_parametrise_regression.py -c ./config/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm christsmas-sweep-1 &
python experiment_parametrise_regression.py -c ./config/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm christsmas-sweep-1 &
#python experiment_parametrise_classification2.py -c ./config/mnist.yaml -ds mnist -ld /scratch/mjh252/logs  -dd /scratch/mjh252/data/UCL

wait
echo "All Finished"