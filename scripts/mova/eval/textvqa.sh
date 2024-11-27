#!/bin/bash

python -m mova.eval.model_vqa_loader \
    --model-path checkpoints/mova-8b \
    --question-file ../CoIN/playground/Instructions_Original/TextVQA/val.json \
    --image-folder ../DatasetCoIN \
    --answers-file ./results/TextVQA/mova-8b.jsonl \
    --temperature 0 \
    --conv-mode mova_llama3

python -m mova.eval.eval_textvqa \
    --annotation-file ../CoIN/playground/Instructions_Original/TextVQA/val.json \
    --result-file ./results/TextVQA/mova-8b.jsonl
