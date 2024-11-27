#!/bin/bash

python -m mova.eval.model_vqa_science \
    --model-path checkpoints/mova-8b \
    --question-file ../CoIN/playground/Instructions_Original/ScienceQA/test.json \
    --image-folder ../DatasetCoIN \
    --answers-file ./results/ScienceQA/mova-8b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode mova_llama3

python mova/eval/eval_science_qa.py \
    --base-dir ../DatasetCoIN/ScienceQA \
    --result-file ./results/ScienceQA/mova-8b.jsonl \
    --output-file ./results/ScienceQA/mova-8b_output.jsonl \
    --output-result ./results/ScienceQA/mova-8b_result.json
