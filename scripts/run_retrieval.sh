#!/bin/bash
set -e

MODEL_PATH=text-embedding-3-small
K=20000
QUERY_CHUNK=10
CORPUS_CHUNK=4
DEVICE=cuda
MAX_TOKEN=1000   # batch size ~1 (avoid NV-Embed OOM on 16GB)
MODEL_TYPE=openai



# query ent to corpus ent

Q_NER_RESP_PATH=output/ner/samples/q_ner_resp.jsonl
Q_ENT_ID_PATH=output/ret/samples/q_ent_id.tsv
C_ENT_ID_PATH=output/ret/samples/c_ent_id.tsv
QE_PATH=output/ret/samples/qe.json
Q_ENT_VECS_PATH=output/vecs/samples/query_vecs.pkl
C_ENT_VECS_PATH=output/vecs/samples/doc_vecs.pkl
E2E_RET_PATH=output/ret/samples/e2e_ret.json

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

QUERY_TSV_PATH=output/ret/samples/samples_query.tsv
DOC_TSV_PATH=output/ret/samples/samples_doc.tsv
QUERY_VECS_PATH=output/vecs/samples/query_vecs.pkl
DOC_VECS_PATH=output/vecs/samples/doc_vecs.pkl
Q2D_RET_PATH=output/ret/samples/q2d_ret.json


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