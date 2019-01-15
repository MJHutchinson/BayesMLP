#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL="3"

export CUDA_VISIBLE_DEVICES=6
python experiment_parametrise_regression.py -c ./config/DGP/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm DGP-Comparison &
python experiment_parametrise_regression.py -c ./config/DGP/concrete.yaml                   -ds concrete                    -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm DGP-Comparison &
python experiment_parametrise_regression.py -c ./config/DGP/bostonHousing.yaml              -ds bostonHousing               -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm DGP-Comparison &
python experiment_parametrise_regression.py -c ./config/DGP/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm DGP-Comparison &

export CUDA_VISIBLE_DEVICES=2
python experiment_parametrise_regression.py -c ./config/DGP/kin8nm.yaml                     -ds kin8nm                      -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm DGP-Comparison &
python experiment_parametrise_regression.py -c ./config/DGP/naval-propulsion-plant.yaml     -ds naval-propulsion-plant      -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm DGP-Comparison &
python experiment_parametrise_regression.py -c ./config/DGP/power-plant.yaml                -ds power-plant                 -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm DGP-Comparison &

export CUDA_VISIBLE_DEVICES=3
python experiment_parametrise_regression.py -c ./config/DGP/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm DGP-Comparison &
#python experiment_parametrise_classification2.py -c ./config/mnist.yaml -ds mnist -ld /scratch/mjh252/logs  -dd /scratch/mjh252/data/UCL
wait
echo "All Finished"