import json
import argparse
from src.evaluation.metrics import calculate_EM, calculate_F1, calculate_recall_scores
from src.utils import ensure_dir

def qa_eval(gold_ans_path, pred_ans_path, save_path=None):
    with open(gold_ans_path, 'r', encoding='utf-8') as f:
        gold_answers = json.load(f)
    with open(pred_ans_path, 'r', encoding='utf-8') as f:
        predicted_answers = json.load(f)
    em, em_v = calculate_EM(gold_answers, predicted_answers)
    f1, f1_v = calculate_F1(gold_answers, predicted_answers)

    if save_path:
        ensure_dir(save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "em": [em] + em_v,
                "f1": [f1] + f1_v
            }, f, ensure_ascii=False, indent=4)

    print('EM:', em)
    print('F1:', f1)
    return em, f1

def cal_recall(gold_docs_id_path, ret_res_path, k_list= [1, 2, 3, 5], save_path=None):
    with open(gold_docs_id_path, 'rb') as f:
        gold_docs = json.load(f)
    with open(ret_res_path, 'rb') as f:
        ret_res = json.load(f)
    ret_docs = [x[0] for x in ret_res.values()]
    recall_scores_all, recall_scores_list = calculate_recall_scores(gold_docs, ret_docs, k_list)
    if save_path:
        ensure_dir(save_path)
        res = [recall_scores_all] + recall_scores_list
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)

    print('Recall@k:', recall_scores_all)
    return recall_scores_all


def main():
    parser = argparse.ArgumentParser(description="Evaluation")

    parser.add_argument(
        "--task",
        required=True,
        choices=["qa_eval", "recall"],
        help="Task to run"
    )

    parser.add_argument("--gold_ans_path", help="Gold answer file path")
    parser.add_argument("--pred_ans_path", help="Predicted answer file path")
    parser.add_argument("--save_path", help="Evaluation results save path")

    parser.add_argument("--gold_docs_id_path", help="Gold docs file")
    parser.add_argument("--ret_res_path", help="Retrieval result file")
    parser.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 5], help="List of k values of Recall@k")

    args = parser.parse_args()

    if args.task == "qa_eval":
        qa_eval(args.gold_ans_path, args.pred_ans_path, args.save_path)

    elif args.task == "recall":
        cal_recall(args.gold_docs_id_path, args.ret_res_path, args.k_list, args.save_path)


if __name__ == "__main__":
    main()

