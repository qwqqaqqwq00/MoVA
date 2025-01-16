# python -m mova.eval.model_vqa \
#     --model-path checkpoints/mova-8b \
#     --question-file ../CoIN/playground/Instructions_Original/ImageNet/test.json \
#     --image-folder ../DatasetCoIN \
#     --answers-file ./playground/data/eval/imagenet/answers/mova-8b.jsonl \
#     --temperature 0 \
#     --conv-mode mova_llama3

# mkdir -p ./playground/data/eval/imagenet/results

python -m mova.eval.eval_ImagetNet \
    --test-file ../CoIN/playground/Instructions_Original/ImageNet/test.json \
    --result-file ./playground/data/eval/imagenet/answers/mova-8b.jsonl \
    --output-dir ./playground/data/eval/imagenet/results \

# python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
#     --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
#     --questions ./playground/Instructions_Original/ImageNet/test.json \
#     --results $output_file \