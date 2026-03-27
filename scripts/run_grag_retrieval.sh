#!/bin/bash
set -e

DATASET_ID_PATH=data/samples/samples_id.json
CORPUS_ID_PATH=data/samples/samples_corpus_id.json

E2E_RET_PATH=output/ret/samples/e2e_ret.json
Q2D_RET_PATH=output/ret/samples/q2d_ret.json
QE_PATH=output/ret/samples/qe.json
GRAG_DOCS_PATH=output/ret/samples/grag_docs.json # save docs

HG_PATH=output/hg/samples/hg.pkl
ENT_DID_PATH=output/hg/samples/c_ent_did.csv

DEVICE=cuda
BETA=0.5
STEP=2
TOPK1=5
TOPK2=10

GRAG_RET_PATH=output/ret/samples/grag_ret.json

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