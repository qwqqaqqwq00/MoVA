#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="mova-8b"
# SPLIT="llava_gqa_testdev_balanced"
GQADIR="../DatasetCoIN/GQA"
STAGE='Finetune'

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m mova.eval.model_vqa_loader \
        --model-path checkpoints/mova-8b \
        --question-file ../CoIN/playground/Instructions_Original/GQA/test.json \
        --image-folder ../DatasetCoIN \
        --answers-file ./results/GQA/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode mova_llama3 &
done

wait

output_file=./results/GQA/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./results/GQA/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst results/GQA/${STAGE}/testdev_balanced_predictions.json

# cd results/GQA
# python eval/eval.py --tier testdev_balanced
# python -m ETrain.Eval.LLaVA.CoIN.eval_gqa --tier testdev_balanced --path $RESULT_DIR/$STAGE --output-dir $RESULT_DIR/$STAGE 
