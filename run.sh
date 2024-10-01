#!/bin/bash


for beta in 1.0 6.0 12.0 
do
    python tot_corr_2layer_nets.py --beta $beta 
done


for layers in 1 3 7
do
    for w in 16 128 512 
    do
        python main.py --num_hidden_layers $layers --width $w
    done
done


