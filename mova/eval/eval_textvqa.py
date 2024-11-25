import os
import argparse
import json
import re

from mova.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    # annotations = json.load(open(annotation_file))['data']
    # annotations = {annotation['question_id']: annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]
    # with open(result_file, 'r') as f:
    #     results = json.load(f)
    
    pred_list = []
    for annotation, result in zip(annotations, results):
        # annotation = annotations[result['question_id']]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": annotation['answer'],
        })
    accs = [1 if a['pred_answer'] == a['gt_answers'] else 0 for a in pred_list]
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * sum(accs)/len(accs)))
    # evaluator = TextVQAAccuracyEvaluator()
    # print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))
