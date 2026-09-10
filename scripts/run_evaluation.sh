#!/bin/bash
set -e

# Usage: bash scripts/run_evaluation.sh [dataset]
# dataset: samples | musique | hotpot | 2wiki  (default: samples)
DATASET="${1:-samples}"

case "$DATASET" in
  samples|musique|hotpot|2wiki) ;;
  *)
    echo "Error: unknown dataset '${DATASET}'" >&2
    echo "Choose: samples | musique | hotpot | 2wiki" >&2
    exit 1
    ;;
esac

GOLD_DOCS_ID_PATH=data/${DATASET}/${DATASET}_gold_docs_id.json
GRAG_RET_PATH=output/ret/${DATASET}/grag_ret.json
SAVE_ER_PATH=output/eval/${DATASET}/eval_rc.json

GOLD_ANS_PATH=data/${DATASET}/${DATASET}_gold_answers.json
PRED_ANS_PATH=output/qa/${DATASET}/qa_pred_ans.json
SAVE_EQA_PATH=output/eval/${DATASET}/eval_qa.json

echo "[evaluation] dataset=${DATASET}"

python -m src.evaluation.evaluator \
    --task recall \
    --gold_docs_id_path "$GOLD_DOCS_ID_PATH" \
    --ret_res_path "$GRAG_RET_PATH" \
    --save_path "$SAVE_ER_PATH"

python -m src.evaluation.evaluator \
    --task qa_eval \
    --gold_ans_path "$GOLD_ANS_PATH" \
    --pred_ans_path "$PRED_ANS_PATH" \
    --save_path "$SAVE_EQA_PATH"
