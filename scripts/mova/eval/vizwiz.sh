#!/bin/bash

# python -m mova.eval.model_vqa_loader \
#     --model-path checkpoints/mova-8b \
#     --question-file ../CoIN/playground/Instructions_Original/VizWiz/val.json \
#     --image-folder ../DatasetCoIN \
#     --answers-file ./results/VizWiz/mova-8b.jsonl \
#     --temperature 0 \
#     --conv-mode mova_llama3

python scripts/convert_vizwiz_for_submission.py \
    --result-file ./results/VizWiz/mova-8b.jsonl \
    --annotation-file ../CoIN/playground/Instructions_Original/VizWiz/val.json \
    --result-upload-file ./results/VizWiz/mova-8b_result.json
