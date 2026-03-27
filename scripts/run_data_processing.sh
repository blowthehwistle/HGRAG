#!/bin/bash
set -e

DATA_DIR=data/samples
DATASET_NAME=musique


CORPUS_PATH=$DATA_DIR/raw/samples_corpus.json
DATASET_PATH=$DATA_DIR/raw/samples.json


CORPUS_ID_PATH=$DATA_DIR/samples_corpus_id.json
DATASET_ID_PATH=$DATA_DIR/samples_id.json


GOLD_DOCS_PATH=$DATA_DIR/samples_gold_docs.json
GOLD_ANS_PATH=$DATA_DIR/samples_gold_answers.json


# for retrieval
QUERY_TSV_PATH=output/ret/samples/samples_query.tsv
DOC_TSV_PATH=output/ret/samples/samples_doc.tsv


python -m src.data_processing \
    --task corpus_add_ids \
    --doc_path "$CORPUS_PATH" \
    --save_path "$CORPUS_ID_PATH"


python -m src.data_processing \
    --task data_add_ids \
    --data_path "$DATASET_PATH" \
    --corpus_id_path "$CORPUS_ID_PATH" \
    --save_path "$DATASET_ID_PATH"


python -m src.data_processing \
    --task get_gold_docs \
    --data_path "$DATASET_ID_PATH" \
    --dataset_name "$DATASET_NAME" \
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