#!/usr/bin/env bash

python parameter_sweep_regression.py -c ./config/DGP/wine-quality-red.yaml              -ds wine-quality-red            -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm DGP_comparison -nd --gpu 0 &
python parameter_sweep_regression.py -c ./config/DGP/bostonHousing.yaml                 -ds bostonHousing               -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm DGP_comparison -nd --gpu 1 &
python parameter_sweep_regression.py -c ./config/DGP/yacht.yaml                         -ds yacht                       -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm DGP_comparison -nd --gpu 2 &
python parameter_sweep_regression.py -c ./config/DGP/protein-tertiary-structure.yaml    -ds protein-tertiary-structure  -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm DGP_comparison -nd --gpu 3 &
python parameter_sweep_regression.py -c ./config/DGP/naval-propulsion-plant.yaml        -ds naval-propulsion-plant      -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm DGP_comparison -nd --gpu 4 &
python parameter_sweep_regression.py -c ./config/DGP/kin8nm.yaml                        -ds kin8nm                      -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm DGP_comparison -nd --gpu 5 &
python parameter_sweep_regression.py -c ./config/DGP/concrete.yaml                      -ds concrete                    -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm DGP_comparison -nd --gpu 6 &
python parameter_sweep_regression.py -c ./config/DGP/power-plant.yaml                   -ds power-plant                 -ld /scratch/mjh252/logs/clean/ -dd /scratch/mjh252/data/UCL/ -cm DGP_comparison -nd --gpu 7 &



wait
echo "All Finished"