#!/bin/bash
set -e

# Usage: bash scripts/run_data_processing.sh [dataset]
# dataset: samples | musique | hotpot | 2wiki  (default: samples)
DATASET="${1:-samples}"

case "$DATASET" in
  samples)
    RAW_STEM=samples
    # samples follow MuSiQue schema
    DATANAME=musique
    GOLD_DATASET_NAME=musique
    ;;
  musique)
    RAW_STEM=musique
    DATANAME=musique
    GOLD_DATASET_NAME=musique
    ;;
  hotpot)
    RAW_STEM=hotpotqa
    DATANAME=hotpotqa
    GOLD_DATASET_NAME=hotpotqa
    ;;
  2wiki)
    RAW_STEM=2wikimultihopqa
    DATANAME=2wikimultihopqa
    GOLD_DATASET_NAME=2wikimultihopqa
    ;;
  *)
    echo "Error: unknown dataset '${DATASET}'" >&2
    echo "Choose: samples | musique | hotpot | 2wiki" >&2
    exit 1
    ;;
esac

DATA_DIR=data/${DATASET}

CORPUS_PATH=$DATA_DIR/raw/${RAW_STEM}_corpus.json
DATASET_PATH=$DATA_DIR/raw/${RAW_STEM}.json

CORPUS_ID_PATH=$DATA_DIR/${DATASET}_corpus_id.json
DATASET_ID_PATH=$DATA_DIR/${DATASET}_id.json

GOLD_DOCS_PATH=$DATA_DIR/${DATASET}_gold_docs.json
GOLD_ANS_PATH=$DATA_DIR/${DATASET}_gold_answers.json

# for retrieval
QUERY_TSV_PATH=output/ret/${DATASET}/${DATASET}_query.tsv
DOC_TSV_PATH=output/ret/${DATASET}/${DATASET}_doc.tsv

echo "[data_processing] dataset=${DATASET}"
echo "  raw qa:     ${DATASET_PATH}"
echo "  raw corpus: ${CORPUS_PATH}"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Error: missing raw dataset file: $DATASET_PATH" >&2
  exit 1
fi
if [[ ! -f "$CORPUS_PATH" ]]; then
  echo "Error: missing raw corpus file: $CORPUS_PATH" >&2
  exit 1
fi

python -m src.data_processing \
    --task corpus_add_ids \
    --doc_path "$CORPUS_PATH" \
    --save_path "$CORPUS_ID_PATH"


python -m src.data_processing \
    --task data_add_ids \
    --data_path "$DATASET_PATH" \
    --corpus_id_path "$CORPUS_ID_PATH" \
    --save_path "$DATASET_ID_PATH" \
    --dataname "$DATANAME"


python -m src.data_processing \
    --task get_gold_docs \
    --data_path "$DATASET_ID_PATH" \
    --dataset_name "$GOLD_DATASET_NAME" \
    --save_path "$GOLD_DOCS_PATH"


python -m src.data_processing \
    --task get_gold_answers \
    --data_path "$DATASET_PATH" \
    --save_path "$GOLD_ANS_PATH"


python -m src.data_processing \
    --task create_query_tsv \
    --data_path "$DATASET_PATH" \
    --save_path "$QUERY_TSV_PATH"


python -m src.data_processing \
    --task create_doc_tsv \
    --data_path "$CORPUS_PATH" \
    --save_path "$DOC_TSV_PATH"
