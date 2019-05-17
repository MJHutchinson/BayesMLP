#!/usr/bin/env bash

#python parameter_sweep_regression.py -c ./config/parameter_sweep_prior_widths/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-prior-var -nd --gpu 4 5 6 7 &
#python parameter_sweep_regression.py -c ./config/parameter_sweep_prior_widths/bostonHousing.yaml              -ds bostonHousing               -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-prior-var -nd --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/parameter_sweep_prior_widths/energy.yaml                     -ds energy                      -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/presplit/ -cm sweep-prior-var -nd --gpu 4 5 6 &

#wait

python parameter_sweep.py -c ./config/parameter_sweep_prior_widths/concrete.yaml                   -ds concrete                    -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-prior-var -nd --gpu 4 5 6 &
#python parameter_sweep_regression.py -c ./config/parameter_sweep_prior_widths/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/UCL/ -cm sweep-prior-var -nd --gpu 4 5 6 7 &

wait

#python parameter_sweep_regression.py -c ./config/parameter_sweep_prior_widths/kin8nm.yaml                     -ds kin8nm                      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-prior-var -nd --gpu 4 5 6 7 &
# python parameter_sweep.py -c ./config/parameter_sweep_prior_widths/naval-propulsion-plant.yaml     -ds naval-propulsion-plant      -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-1 --gpu 4 5 6 7 &
python parameter_sweep.py -c ./config/parameter_sweep_prior_widths/power-plant.yaml                -ds power-plant                 -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-prior-var -nd --gpu 4 5 6 &

#wait

python parameter_sweep.py -c ./config/parameter_sweep_prior_widths/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs/clean -dd /scratch/mjh252/data/presplit/ -cm sweep-prior-var -nd --gpu 4 5 6 &
#python experiment_parametrise_classification2.py -c ./config/mnist.yaml -ds mnist -ld /scratch/mjh252/logs  -dd /scratch/mjh252/data/UCL

wait
echo "All Finished"