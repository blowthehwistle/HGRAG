#!/bin/bash
set -e

# Usage: bash scripts/run_retrieval.sh [dataset]
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

MODEL_PATH=text-embedding-3-small
K=20000
QUERY_CHUNK=10
CORPUS_CHUNK=4
DEVICE=cuda
MAX_TOKEN=1000
MODEL_TYPE=openai

# query ent to corpus ent
Q_NER_RESP_PATH=output/ner/${DATASET}/q_ner_resp.jsonl
Q_ENT_ID_PATH=output/ret/${DATASET}/q_ent_id.tsv
C_ENT_ID_PATH=output/ret/${DATASET}/c_ent_id.tsv
QE_PATH=output/ret/${DATASET}/qe.json
Q_ENT_VECS_PATH=output/vecs/${DATASET}/q_ent_vecs.pkl
C_ENT_VECS_PATH=output/vecs/${DATASET}/c_ent_vecs.pkl
E2E_RET_PATH=output/ret/${DATASET}/e2e_ret.json

# query to doc
QUERY_TSV_PATH=output/ret/${DATASET}/${DATASET}_query.tsv
DOC_TSV_PATH=output/ret/${DATASET}/${DATASET}_doc.tsv
QUERY_VECS_PATH=output/vecs/${DATASET}/query_vecs.pkl
DOC_VECS_PATH=output/vecs/${DATASET}/doc_vecs.pkl
Q2D_RET_PATH=output/ret/${DATASET}/q2d_ret.json

echo "[retrieval] dataset=${DATASET} model=${MODEL_PATH} type=${MODEL_TYPE}"

if [[ ! -f "$Q_NER_RESP_PATH" ]]; then
  echo "Error: missing ${Q_NER_RESP_PATH}" >&2
  exit 1
fi
if [[ ! -f "$C_ENT_ID_PATH" ]]; then
  echo "Error: missing ${C_ENT_ID_PATH}. Run: bash scripts/run_build_hgraph.sh ${DATASET}" >&2
  exit 1
fi
if [[ ! -f "$QUERY_TSV_PATH" || ! -f "$DOC_TSV_PATH" ]]; then
  echo "Error: missing query/doc tsv. Run: bash scripts/run_data_processing.sh ${DATASET}" >&2
  exit 1
fi

python -m src.data_processing \
    --task qner_res_process \
    --data_path "$Q_NER_RESP_PATH" \
    --save_path "$Q_ENT_ID_PATH" \
    --save_qe_path "$QE_PATH"


# q_ent to doc_ent
python -m src.retrieval \
    --model_path "$MODEL_PATH" \
    --corpus_path "$C_ENT_ID_PATH" \
    --query_path "$Q_ENT_ID_PATH" \
    --corpus_vecs_path "$C_ENT_VECS_PATH" \
    --query_vecs_path "$Q_ENT_VECS_PATH" \
    --retrival_res_path "$E2E_RET_PATH" \
    --k "$K" \
    --query_chunk "$QUERY_CHUNK" \
    --corpus_chunk "$CORPUS_CHUNK" \
    --device "$DEVICE" \
    --max_token "$MAX_TOKEN" \
    --model_type "$MODEL_TYPE" \
    --reindex


# query to doc
python -m src.retrieval \
    --model_path "$MODEL_PATH" \
    --corpus_path "$DOC_TSV_PATH" \
    --query_path "$QUERY_TSV_PATH" \
    --corpus_vecs_path "$DOC_VECS_PATH" \
    --query_vecs_path "$QUERY_VECS_PATH" \
    --retrival_res_path "$Q2D_RET_PATH" \
    --k "$K" \
    --query_chunk "$QUERY_CHUNK" \
    --corpus_chunk "$CORPUS_CHUNK" \
    --device "$DEVICE" \
    --max_token "$MAX_TOKEN" \
    --prompt "query2passages" \
    --model_type "$MODEL_TYPE" \
    --reindex
