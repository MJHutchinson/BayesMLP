#!/usr/bin/env bash

python parameter_sweep_regression.py -c ./config/weight_pruning/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning -nd --gpu 0 1 2 3 &
python parameter_sweep_regression.py -c ./config/weight_pruning/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning -nd --gpu 0 1 2 3 &
python parameter_sweep_regression.py -c ./config/weight_pruning/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs/ -dd /scratch/mjh252/data/UCL/ -cm weight_pruning -nd --gpu 5 6 7 &

wait
echo "All Finished"