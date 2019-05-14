#!/usr/bin/env bash

#python parameter_sweep.py -c ./config/parameter_sweep_layer_sizes/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs/clean  -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd --gpu 4 5 6 7 &
#python parameter_sweep.py -c ./config/parameter_sweep_layer_sizes/bostonHousing.yaml              -ds bostonHousing               -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/parameter_sweep_layer_sizes/energy.yaml                     -ds energy                      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-hidden-sizes -nd --gpu 4 5 6 7 &

wait

python parameter_sweep.py -c ./config/parameter_sweep_layer_sizes/concrete.yaml                   -ds concrete                    -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-hidden-sizes -nd --gpu 4 5 6 7 &
#python parameter_sweep.py -c ./config/parameter_sweep_layer_sizes/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd --gpu 4 5 6 7 &

wait

#python parameter_sweep.py -c ./config/parameter_sweep_layer_sizes/kin8nm.yaml                     -ds kin8nm                      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd --gpu 4 5 6 7 &
# python parameter_sweep.py -c ./config/parameter_sweep_layer_sizes/naval-propulsion-plant.yaml     -ds naval-propulsion-plant      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/parameter_sweep_layer_sizes/power-plant.yaml                -ds power-plant                 -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-hidden-sizes -nd --gpu 4 5 6 7 &

wait

python parameter_sweep.py -c ./config/parameter_sweep_layer_sizes/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-hidden-sizes -nd --gpu 4 5 6 7 &
#python experiment_parametrise_classification2.py -c ./config/parameter_sweep_layer_sizes/mnist.yaml -ds mnist -ld /scratch/mjh252/logs  -dd /scratch/mjh252/data/UCL

wait
echo "All Finished"