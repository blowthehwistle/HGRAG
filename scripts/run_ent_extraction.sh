#!/bin/bash
set -e

# Usage: bash scripts/run_ent_extraction.sh [dataset]
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
DATA_PATH=data/${DATASET}/${DATASET}_id.json
CORPUS_PATH=data/${DATASET}/${DATASET}_corpus_id.json
Q_NER_RESP_PATH=output/ner/${DATASET}/q_ner_resp.jsonl
C_NER_RESP_PATH=output/ner/${DATASET}/c_ner_resp.jsonl
LOG_PATH=logs/ent_extract_${DATASET}.log

MAX_BATCH_TOKENS=1000
MAX_NEW_TOKENS=1000
DEVICE="auto"

echo "[ent_extraction] dataset=${DATASET}"
echo "  query: ${DATA_PATH}"
echo "  corpus: ${CORPUS_PATH}"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Error: missing ${DATA_PATH}. Run: bash scripts/run_data_processing.sh ${DATASET}" >&2
  exit 1
fi
if [[ ! -f "$CORPUS_PATH" ]]; then
  echo "Error: missing ${CORPUS_PATH}. Run: bash scripts/run_data_processing.sh ${DATASET}" >&2
  exit 1
fi

# ent extraction for query
python -m src.ent_extraction \
  --model_id "$MODEL_ID" \
  --data_path "$DATA_PATH" \
  --resp_path "$Q_NER_RESP_PATH" \
  --type "Query" \
  --max_batch_tokens "$MAX_BATCH_TOKENS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --log_path "$LOG_PATH"

# ent extraction for corpus
python -m src.ent_extraction \
  --model_id "$MODEL_ID" \
  --data_path "$CORPUS_PATH" \
  --resp_path "$C_NER_RESP_PATH" \
  --type "Corpus" \
  --max_batch_tokens "$MAX_BATCH_TOKENS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --log_path "$LOG_PATH"
