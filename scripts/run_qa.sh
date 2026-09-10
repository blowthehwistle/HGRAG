#!/bin/bash
set -e

# Usage: bash scripts/run_qa.sh [dataset]
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

MODEL_ID=gpt-4o-mini
GRAG_DOCS_PATH=output/ret/${DATASET}/grag_docs.json
RESP_PATH=output/qa/${DATASET}/qa_resp.jsonl
PRED_ANS_PATH=output/qa/${DATASET}/qa_pred_ans.json

MAX_BATCH_TOKENS=1000
MAX_NEW_TOKENS=1000
DEVICE=auto
LOG_PATH=logs/qa_${DATASET}.log

echo "[qa] dataset=${DATASET} model=${MODEL_ID}"

if [[ ! -f "$GRAG_DOCS_PATH" ]]; then
  echo "Error: missing ${GRAG_DOCS_PATH}. Run: bash scripts/run_grag_retrieval.sh ${DATASET}" >&2
  exit 1
fi

# append mode in Inferencer — wipe previous run for this dataset
rm -f "$RESP_PATH"

python -m src.qa \
  --model_id "$MODEL_ID" \
  --data_path "$GRAG_DOCS_PATH" \
  --resp_path "$RESP_PATH" \
  --max_batch_tokens "$MAX_BATCH_TOKENS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --log_path "$LOG_PATH"

python -m src.data_processing \
    --task qa_res_process \
    --data_path "$RESP_PATH" \
    --save_path "$PRED_ANS_PATH"
