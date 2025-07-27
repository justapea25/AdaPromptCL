#!/bin/bash

# AdaPromptCL Training Script for Deepfake Dataset
# Example 1: Train on 2 specific tasks (biggan and gaugan) with AdaPromptCL features

python main.py \
    deepfake_dualprompt \
    --model vit_base_patch16_224 \
    --batch-size 24 \
    --data-path /home/hoangminhdau/work/Working/DynaCon/data/processed/CDDB \
    --output_dir ./output/deepfake_adapromptcl \
    --epochs 2 \
    --dataset Split-Deepfake \
    --num_tasks 2 \
    --deepfake_tasks biggan gaugan \
    --binary_evaluation \
    --use_e_prompt --e_prompt_layer_idx 0 1 2 3 4 \
    --data_driven_evolve --uni_or_specific --converge_thrh 0.4 \
    --mergable_prompts --postmerge_thrh 0.6

# Example 2: Train on 3 tasks with more aggressive merging (uncomment to use)
# python main.py \
#     deepfake_dualprompt \
#     --model vit_base_patch16_224 \
#     --batch-size 24 \
#     --data-path /home/hoangminhdau/work/Working/DynaCon/data/processed/CDDB \
#     --output_dir ./output/deepfake_adapromptcl_3tasks \
#     --epochs 5 \
#     --num_tasks 3 \
#     --deepfake_tasks stylegan gaugan crn \
#     --binary_evaluation \
#     --class_num_binary 2 \
#     --train_mask \
#     --use_e_prompt --e_prompt_layer_idx 0 1 2 3 4 \
#     --data_driven_evolve --uni_or_specific --converge_thrh 0.3 \
#     --mergable_prompts --postmerge_thrh 0.5 