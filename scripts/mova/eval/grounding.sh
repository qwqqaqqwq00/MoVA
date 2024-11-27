#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# gpu_list="4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/mova-8b'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/Grounding"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m mova.eval.model_vqa \
        --model-path $MODELPATH \
        --question-file ../CoIN/playground/Instructions_Original/Grounding/test.json \
        --image-folder ../DatasetCoIN \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode mova_llama3 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m mova.eval.eval_grounding \
    --test-file ../CoIN/playground/Instructions_Original/Grounding/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

python -m mova.eval.create_prompt \
    --rule ../CoIN/ETrain/Eval/LLaVA/CoIN/rule.json \
    --questions ../CoIN/playground/Instructions_Original/Grounding/test.json \
    --results $output_file \
    --rule_temp CoIN_Grounding \