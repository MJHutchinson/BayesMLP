#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
python parameter_sweep_regression.py -c ./config/parameter_sweep_layer_sizes/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs/clean  -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd &
python parameter_sweep_regression.py -c ./config/parameter_sweep_layer_sizes/bostonHousing.yaml              -ds bostonHousing               -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd &
#python parameter_sweep_regression.py -c ./config/parameter_sweep_1/energy.yaml                     -ds energy                      -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-2-prior-var &

export CUDA_VISIBLE_DEVICES=3
#python parameter_sweep_regression.py -c ./config/parameter_sweep_layer_sizes/concrete.yaml                   -ds concrete                    -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd &
python parameter_sweep_regression.py -c ./config/parameter_sweep_layer_sizes/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd &

export CUDA_VISIBLE_DEVICES=3
python parameter_sweep_regression.py -c ./config/parameter_sweep_layer_sizes/kin8nm.yaml                     -ds kin8nm                      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd &
# python parameter_sweep_regression.py -c ./config/parameter_sweep_layer_sizes/naval-propulsion-plant.yaml     -ds naval-propulsion-plant      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes &
#python parameter_sweep_regression.py -c ./config/parameter_sweep_layer_sizes/power-plant.yaml                -ds power-plant                 -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd &

export CUDA_VISIBLE_DEVICES=4
#python parameter_sweep_regression.py -c ./config/parameter_sweep_layer_sizes/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-hidden-sizes -nd &
#python experiment_parametrise_classification2.py -c ./config/parameter_sweep_layer_sizes/mnist.yaml -ds mnist -ld /scratch/mjh252/logs  -dd /scratch/mjh252/data/UCL
wait
echo "All Finished"