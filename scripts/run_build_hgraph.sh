#!/bin/bash
set -e

# Usage: bash scripts/run_build_hgraph.sh [dataset]
# dataset: samples | musique | hotpot | 2wiki  (default: samples)
DATASET="${1:-samples}"

C_NER_RESP_PATH=output/ner/${DATASET}/c_ner_resp.jsonl
C_ENT_DID_PATH=output/hg/${DATASET}/c_ent_did.csv

HG_PATH=output/hg/${DATASET}/hg.pkl
C_ENT_ID_PATH=output/ret/${DATASET}/c_ent_id.tsv # for retrieval

echo "[build_hgraph] dataset=${DATASET}"
echo "  ner:  ${C_NER_RESP_PATH}"
echo "  hg:   ${HG_PATH}"

if [[ ! -f "$C_NER_RESP_PATH" ]]; then
  echo "Error: CNER result not found: $C_NER_RESP_PATH" >&2
  echo "Run entity extraction for '${DATASET}' first." >&2
  exit 1
fi

#generate ent-docid mapping csv file from ner results for hypergraph creation
python -m src.data_processing \
    --task cner_res_process \
    --data_path "$C_NER_RESP_PATH" \
    --save_path "$C_ENT_DID_PATH"

python -m src.build_hgraph \
    --ent_did_path "$C_ENT_DID_PATH" \
    --hg_save_path "$HG_PATH" \
    --ent_id_save_path "$C_ENT_ID_PATH"
