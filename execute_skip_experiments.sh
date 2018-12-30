#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL="3"

export CUDA_VISIBLE_DEVICES=1
python experiment_parametrise_skip_connect.py -c ./config/skips/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm skips &
export CUDA_VISIBLE_DEVICES=2
python experiment_parametrise_skip_connect.py -c ./config/skips/bostonHousing.yaml              -ds bostonHousing               -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm skips &
export CUDA_VISIBLE_DEVICES=3
python experiment_parametrise_skip_connect.py -c ./config/skips/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm skips &
export CUDA_VISIBLE_DEVICES=4
python experiment_parametrise_skip_connect.py -c ./config/skips/concrete.yaml                   -ds concrete                    -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm skips &
export CUDA_VISIBLE_DEVICES=5
python experiment_parametrise_skip_connect.py -c ./config/skips/kin8nm.yaml                     -ds kin8nm                      -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm skips &
export CUDA_VISIBLE_DEVICES=6
python experiment_parametrise_skip_connect.py -c ./config/skips/naval-propulsion-plant.yaml     -ds naval-propulsion-plant      -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm skips &
export CUDA_VISIBLE_DEVICES=1
python experiment_parametrise_skip_connect.py -c ./config/skips/power-plant.yaml                -ds power-plant                 -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm skips &
export CUDA_VISIBLE_DEVICES=2
python experiment_parametrise_skip_connect.py -c ./config/skips/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs -dd /scratch/mjh252/data/UCL/ -cm skips &
#export CUDA_VISIBLE_DEVICES=3
#python experiment_parametrise_classification2.py -c ./config/mnist.yaml -ds mnist -ld /scratch/mjh252/logs  -dd /scratch/mjh252/data/UCL
wait
echo "All Finished"