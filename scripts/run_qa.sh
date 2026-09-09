#!/bin/bash
set -e

MODEL_ID=gpt-4o-mini
GRAG_DOCS_PATH=output/ret/samples/grag_docs.json
RESP_PATH=output/qa/samples/qa_resp.jsonl

MAX_BATCH_TOKENS=1000
MAX_NEW_TOKENS=1000
DEVICE=auto

LOG_PATH=logs/qa.log

python -m src.qa \
  --model_id "$MODEL_ID" \
  --data_path "$GRAG_DOCS_PATH" \
  --resp_path "$RESP_PATH" \
  --max_batch_tokens "$MAX_BATCH_TOKENS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --log_path "$LOG_PATH"


PRED_ANS_PATH=output/qa/samples/qa_pred_ans.json

python -m src.data_processing \
    --task qa_res_process \
    --data_path "$RESP_PATH" \
    --save_path "$PRED_ANS_PATH"