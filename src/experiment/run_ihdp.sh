#!/usr/bin/env bash

# Set up the Python path to include src directory (fix for running from bash)
export PYTHONPATH="G:/My Drive/CML_Research/Practice Code/dragonnet/src"

options=(
    dragonnet
    #tarnet

)



for i in ${options[@]}; do
    echo $i
    python -m experiment.ihdp_main --data_base_dir dat/ihdp/csv\
                                 --knob $i\
                                 --output_base_dir co_implement/ihdp\


done