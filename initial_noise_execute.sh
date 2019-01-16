#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
python data_multiply.py -c ./config/initial_noise/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm initial-noise -nd &
#python data_multiply.py -c ./config/initial_noise/energy.yaml                     -ds energy                      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm initial-noise -nd &

export CUDA_VISIBLE_DEVICES=1
#python data_multiply.py -c ./config/initial_noise/concrete.yaml                   -ds concrete                    -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm initial-noise -nd &
python data_multiply.py -c ./config/initial_noise/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm initial-noise -nd &

export CUDA_VISIBLE_DEVICES=4
python data_multiply.py -c ./config/initial_noise/bostonHousing.yaml              -ds bostonHousing               -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm initial-noise -nd &

#python data_multiply.py -c ./config/initial_noise/kin8nm.yaml                     -ds kin8nm                      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm initial-noise -nd &
# python parameter_sweep_regression.py -c ./config/initial_noise/naval-propulsion-plant.yaml     -ds naval-propulsion-plant      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm initial-noise &
#python data_multiply.py -c ./config/initial_noise/power-plant.yaml                -ds power-plant                 -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm initial-noise -nd &

export CUDA_VISIBLE_DEVICES=4
#python data_multiply.py -c ./config/initial_noise/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm initial-noise -nd &
#python experiment_parametrise_classification2.py -c ./config/initial_noise/mnist.yaml -ds mnist -ld /scratch/mjh252/logs  -dd /scratch/mjh252/data/UCL

wait
echo "All Finished"