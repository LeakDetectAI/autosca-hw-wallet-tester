#!/bin/bash

# Default settings
DATASETS=("KEYSTROKE" "PIN_MOTION")
SEEDS=(1234 42 100 999 2024)
EPOCHS=30
SEARCH="random"

echo "========================================================="
echo "Automating NAS runs across multiple seeds"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "========================================================="

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "---------------------------------------------------------"
        echo "Running Domain: BEHAVIORAL | Dataset: $dataset with seed $seed"
        echo "---------------------------------------------------------"
        /opt/homebrew/Caskroom/miniforge/base/envs/autosca/bin/python exp_behavioral_run.py --domain BEHAVIORAL --dataset $dataset --search $SEARCH --epochs $EPOCHS --seed $seed
    done
done

echo "========================================================="
echo "All runs completed successfully."
echo "========================================================="
