#!/bin/bash
set -e

# Usage: bash scripts/run_grag_retrieval.sh [dataset]
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

DATASET_ID_PATH=data/${DATASET}/${DATASET}_id.json
CORPUS_ID_PATH=data/${DATASET}/${DATASET}_corpus_id.json

E2E_RET_PATH=output/ret/${DATASET}/e2e_ret.json
Q2D_RET_PATH=output/ret/${DATASET}/q2d_ret.json
QE_PATH=output/ret/${DATASET}/qe.json
GRAG_DOCS_PATH=output/ret/${DATASET}/grag_docs.json

HG_PATH=output/hg/${DATASET}/hg.pkl
ENT_DID_PATH=output/hg/${DATASET}/c_ent_did.csv

DEVICE=cuda
BETA=0.5
STEP=2
TOPK1=5
TOPK2=10

GRAG_RET_PATH=output/ret/${DATASET}/grag_ret.json

echo "[grag_retrieval] dataset=${DATASET}"

python -m src.grag_retrieval \
  --data_path "$DATASET_ID_PATH" \
  --corpus_path "$CORPUS_ID_PATH" \
  --e2e_ret_path "$E2E_RET_PATH" \
  --q2d_ret_path "$Q2D_RET_PATH" \
  --qe_path "$QE_PATH" \
  --grag_docs_path "$GRAG_DOCS_PATH" \
  --hg_path "$HG_PATH" \
  --ent_did_path "$ENT_DID_PATH" \
  --device "$DEVICE" \
  --beta "$BETA" \
  --step "$STEP" \
  --topk1 "$TOPK1" \
  --topk2 "$TOPK2" \
  --grag_ret_path "$GRAG_RET_PATH"
