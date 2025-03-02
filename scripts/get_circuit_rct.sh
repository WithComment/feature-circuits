#!/bin/bash

MODEL=EleutherAI/pythia-70m-deduped
DATA=prompts/concise
NODE=$3
EDGE=$4
AGG=$5

python circuit.py \
    --model $MODEL \
    --num_examples 100 \
    --batch_size 10 \
    --prompt_dir $DATA \
	--node_threshold $NODE \
	--edge_threshold $EDGE \
	--aggregation $AGG