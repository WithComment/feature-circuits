#!/bin/bash

MODEL=EleutherAI/pythia-70m-deduped
CIRCUIT=circuits/pythia-70m-deduped_simple_train_n20_aggsum_node0.1.pt
EVAL_DATA=prompts/concise
THRESHOLD=0.1
START_LAYER=2


# Run the ablation.py script with the specified arguments
python ablation.py \
--model $MODEL \
--circuit $CIRCUIT \
--data ${EVAL_DATA}.json \
--examples 40 \
--threshold $THRESHOLD \
--ablation resample \
--handle_errors 'default' \
--start_layer $START_LAYER \
--device cuda:0