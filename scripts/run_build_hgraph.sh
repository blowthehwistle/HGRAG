#!/bin/bash
set -e

C_NER_RESP_PATH=output/ner/samples/c_ner_resp.jsonl
C_ENT_DID_PATH=output/hg/samples/c_ent_did.csv

HG_PATH=output/hg/samples/hg.pkl
C_ENT_ID_PATH=output/ret/samples/c_ent_id.tsv # for retrieval

#generate ent-docid mapping csv file from ner results for hypergraph creation
python -m src.data_processing \
    --task cner_res_process \
    --data_path "$C_NER_RESP_PATH" \
    --save_path "$C_ENT_DID_PATH"

python -m src.build_hgraph \
    --ent_did_path "$C_ENT_DID_PATH" \
    --hg_save_path "$HG_PATH" \
    --ent_id_save_path "$C_ENT_ID_PATH"


