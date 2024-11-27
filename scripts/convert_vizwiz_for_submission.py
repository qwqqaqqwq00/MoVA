import os
import argparse
import json
import sys
sys.path = [path for path in sys.path if '/root/MoVA/scripts/mova/eval' not in path]  # 移除错误路径
sys.path.insert(0, '/root/MoVA')
# from MoVA.mova.eval.m4c_evaluator import EvalAIAnswerProcessor
from mova.eval.m4c_evaluator import EvalAIAnswerProcessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, required=True)
    parser.add_argument('--result-file', type=str, required=True)
    parser.add_argument('--result-upload-file', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    os.makedirs(os.path.dirname(args.result_upload_file), exist_ok=True)

    results = []
    error_line = 0
    for line_idx, line in enumerate(open(args.result_file)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1
    # results = {x['question_id']: x['text'] for x in results}
    # test_split = [json.loads(line) for line in open(args.annotation_file)]
    test_split = json.load(open(args.annotation_file, 'r'))
    split_ids = set([x['question_id'] for x in test_split])
    
    pred_list = []
    for annotation, result in zip(test_split, results):
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": annotation['answer'],
        })
    accs = [1 if a['pred_answer'] == a['gt_answers'] else 0 for a in pred_list]
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * sum(accs)/len(accs)))

    print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

    all_answers = []
    
    answer_processor = EvalAIAnswerProcessor()

    for x, res in zip(test_split, results):
        # assert x['question_id'] in res
        all_answers.append({
            'image': x['image'],
            'answer': answer_processor(res['text'])
        })

    with open(args.result_upload_file, 'w') as f:
        json.dump(all_answers, f)
