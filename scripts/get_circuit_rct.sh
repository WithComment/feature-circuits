#!/bin/bash

MODEL=EleutherAI/pythia-70m-deduped
DATA=prompts/concise
NODE=$3
EDGE=$4
AGG=$5
METHOD=$6

python3 circuit.py \
    --model $MODEL \
    --num_examples 2 \
    --batch_size 10 \
    --prompt_dir $DATA \
	--node_threshold $NODE \
	--edge_threshold $EDGE \
	--aggregation $AGG \
    --method $METHOD
    