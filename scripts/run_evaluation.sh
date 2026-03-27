#!/bin/bash
set -e

GOLD_DOCS_ID_PATH=data/samples/samples_gold_docs_id.json
GRAG_RET_PATH=output/ret/samples/grag_ret.json
SAVE_ER_PATH=output/eval/samples/eval_rc.json

python -m src.evaluation.evaluator \
    --task recall \
    --gold_docs_id_path "$GOLD_DOCS_ID_PATH" \
    --ret_res_path "$GRAG_RET_PATH" \
    --save_path "$SAVE_ER_PATH"



GOLD_ANS_PATH=data/samples/samples_gold_answers.json
PRED_ANS_PATH=output/qa/samples/qa_pred_ans.json
SAVE_EQA_PATH=output/eval/samples/eval_qa.json

python -m src.evaluation.evaluator \
    --task qa_eval \
    --gold_ans_path "$GOLD_ANS_PATH" \
    --pred_ans_path "$PRED_ANS_PATH" \
    --save_path "$SAVE_EQA_PATH"