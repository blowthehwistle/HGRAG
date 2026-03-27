#!/bin/bash
set -e

MODEL_ID=/home/user/llms/Llama-3.3-70B-Instruct
DATA_PATH=data/samples/samples_id.json
CORPUS_PATH=data/samples/samples_corpus_id.json
Q_NER_RESP_PATH=output/ner/samples/q_ner_resp.jsonl
C_NER_RESP_PATH=output/ner/samples/c_ner_resp.jsonl
LOG_PATH=logs/ent_extract.log

MAX_BATCH_TOKENS=1000
MAX_NEW_TOKENS=1000
DEVICE="auto"

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


