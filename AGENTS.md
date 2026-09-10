# HGRAG — Agent Guide

Code for **Cross-Granularity Hypergraph Retrieval-Augmented Generation for
Multi-hop Question Answering** (Wang et al., AAAI 2026, arXiv:2508.11247).

Benchmarks and several prompts follow [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG).
File-by-file map: [`CODEBASE.md`](./CODEBASE.md).

This checkout is **not** a bit-identical paper reproduction. NER/QA already use
OpenAI `gpt-4o-mini`. The paper used Llama-3.3-70B-Instruct. Dense retrieval still
targets local **NV-Embed-v2**.

---

## What the method does

HGRAG builds an **entity hypergraph**: entities are nodes, passages are
hyperedges. Shared entities connect passages. At query time it fuses two dense
similarities via hypergraph diffusion, then optionally expands by hyperedge
adjacency.

```
corpus / questions
        │
        ▼
 data_processing   IDs, gold docs/answers, query/doc TSVs
        │
        ▼
 ent_extraction    LLM NER  →  query entities + corpus entities
        │
        ├──────────────────────────┐
        ▼                          ▼
 build_hgraph                 retrieval
 entity–doc incidence          (1) e2e: query-ent ↔ corpus-ent
 H, De, Dv                     (2) q2d: query text ↔ passages
        │                          │
        └──────────┬───────────────┘
                   ▼
            grag_retrieval
            seed ents + qd sims → diffusion + structural enhance
                   │
                   ▼
                  qa  →  evaluation (Recall@k, EM, F1)
```

The paper's **6× retrieval-efficiency** claim vs HippoRAG 2 is about *online
graph retrieval* (smaller graph, no PPR over a dense fact graph), not about
embedding-model FLOPs. Embeddings are precomputed.

---

## Pipeline (run from `HGRAG/`)

```bash
bash scripts/run_data_processing.sh
bash scripts/run_ent_extraction.sh   # needs OPENAI_API_KEY in .env
bash scripts/run_build_hgraph.sh
bash scripts/run_retrieval.sh        # needs NV-Embed-v2 on GPU  ← current blocker
bash scripts/run_grag_retrieval.sh
bash scripts/run_qa.sh
bash scripts/run_evaluation.sh
```

Sample data: `data/samples/raw/` (2 MuSiQue-style questions, ~40 passages).
Full HippoRAG-format dumps live under `data/{musique,hotpot,2wiki}/raw/`.

Scripts are hard-coded to `samples`. To run a real dataset, copy a script and
rewire paths; there is no `--dataset` switch.

---

## Retrieval internals (what efficiency experiments touch)

### Dense retrieval — `src/modules/retriever.py`

| Class | When | How |
|-------|------|-----|
| `NVERetriever` | `--model_type nve` (default) | `AutoModel(..., trust_remote_code=True, device_map="auto")`, then `plm.encode`. Query side prepends `Instruct: {prompt}\nQuery: `. |
| `Retriever` | `--model_type contriever` | HF `AutoModel` + mean pool over last hidden state. |
| `ColbRetriever` | unused | ColBERTv2 index/search. |

Vectors are L2-normalized; search is cosine via `torch.matmul` (default) or FAISS
GPU `IndexFlat` IP. Output JSON:

```json
{"qid": [[did, ...], [score, ...]]}
```

`src/retrieval.py` writes pickle dumps `{s2v, vecs}` then runs search. `k` in
`run_retrieval.sh` is `20000` so q2d can later look up **every** passage score
(needed as the diagonal weight `Wp` in diffusion).

NV-Embed instruction keys live in `PROMPTS["NVEmd"]`. q2d uses `query2passages`.
e2e currently passes `prompt=None`, so query entities are encoded **without** the
`ner2node` instruction the paper-style prompts define.

### Hypergraph — `src/modules/hgraph.py`

From `ent,did` CSV:

- `H` sparse incidence `|E| × |D|`
- `De_inv`, `Dv_inv_sqrt` from hyperedge / vertex degrees
- `hg_diffusion(ent_ids, ent_sims, qd_sims, beta, step, multihot)`:
  - `X` = multi-hot seed entities (default) or max-pooled e2e scores
  - `Wp = diag(qd_sims)` — missing docs get whatever the caller put in `dsim`
  - `L = Dv^{-1/2} H Wp De^{-1} H^T Dv^{-1/2}`
  - `P = β · p + (1-β) · Wp (H^T L^{step} X)`
- `struct_enhance`: keep top-`k1`, then from ranks `k1..k2` keep only passages
  adjacent in `H^T H` (share at least one entity) to the top-`k1` set.

Paper defaults used here: `beta=0.5`, `step=2`, `topk1=5`, `topk2=10`.

### Seed construction — `src/modules/grag.py`

For each query entity, take top-`ent_topk` (default 1) corpus entities from e2e,
optional `ent_thr` cutoff, concatenate into seed ids/scores, attach the full q2d
score map, then run diffusion.

---

## Embedding models across related systems

### This paper (HGRAG)

- Encoder **E**: NVIDIA **NV-Embed-v2** (7B, 4096-d, latent-attention pooling,
  max length 32768; this code caps `max_length=1024` to limit VRAM).
- LLM (paper): Llama-3.3-70B-Instruct, temperature 0, for NER and QA.
- This fork LLM: `gpt-4o-mini` via `src/modules/inferencer.py`.

NV-Embed-v2 is a Mistral-7B embedder. On an RTX 5060 Ti it does not load in a
usable way. `scripts/run_retrieval.sh` already uses `MAX_TOKEN=1` ("avoid
NV-Embed OOM on 16GB") and a local snapshot:

`/media/hwi/Data/huggingface/hub/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef`

### HippoRAG

| Version | Paper encoder | Code |
|---------|---------------|------|
| HippoRAG 1 | Contriever | `legacy` branch |
| HippoRAG 2 | `nvidia/NV-Embed-v2` | Default `embedding_model_name="nvidia/NV-Embed-v2"`. Router also accepts Contriever, GritLM, OpenAI `text-embedding-*`, Cohere, Transformers, vLLM. |

HippoRAG's own demo for a no-GPU / OpenAI-only box is:

```python
HippoRAG(..., llm_model_name="gpt-4o-mini", embedding_model_name="text-embedding-3-small")
```

Paper-reproduction `main.py` still passes `--embedding_name nvidia/NV-Embed-v2`.

### HyperGraphRAG (this workspace)

Paper + `hypergraphrag/hypergraphrag.py` default:

- `embedding_func = openai_embedding`
- model `text-embedding-3-small`, dim 1536
- LLM `gpt-4o-mini`

See `HyperGraphRAG/hypergraphrag/llm.py` (`openai_embedding`, also Azure / NVIDIA
NIM / Jina / HF / Ollama).

### Practical choice on this GPU

| Option | VRAM | Paper-faithful for HGRAG / HippoRAG 2 | Already in HGRAG code |
|--------|------|----------------------------------------|------------------------|
| NV-Embed-v2 local | too high | yes | yes (`nve`) |
| Contriever local | low | HippoRAG 1 only | yes (`contriever`) |
| OpenAI `text-embedding-3-small` | none | HyperGraphRAG paper; HippoRAG-supported, not HGRAG paper | **no — must add a retriever** |
| OpenAI `text-embedding-3-large` | none | neither paper | no |

For **retrieval-efficiency** (index size, diffusion vs PPR wall time, Recall@k
vs retrieved-context tokens), the encoder only needs to be **shared and frozen**.
`text-embedding-3-small` is the least-friction choice and matches HyperGraphRAG.
Do not compare those Recall numbers to the HGRAG AAAI table (that table is
NV-Embed-v2).

If OpenAI embeddings are added, keep the same contract as `Retriever.indexing`:
TSV `id\t"text"` in, pickle `{s2v, vecs}` out, L2-normalize before IP search.
Cache under distinct paths for e2e vs q2d (see known issues).

---

## Known issues / traps

1. **`run_retrieval.sh` reuses the same pickle paths** for entity vectors and
   passage vectors (`output/vecs/samples/query_vecs.pkl`, `doc_vecs.pkl`). The
   q2d stage overwrites the e2e dumps. e2e JSON is already written, so the
   sample pipeline still works, but you cannot resume or inspect entity vectors
   afterward. Split the paths before any real experiment.

2. **q2d `k=20000`**. `HG._get_Wp` indexes `qd_sims[did]` for every `did` in
   `0..doc_num-1`. If a doc is missing from the q2d map this will KeyError. Keep
   `k >= corpus size` or change `_get_Wp` to default missing scores to 0.

3. **`NVERetriever` has no tokenizer**. `_split_strs(..., 'None')` batches by
   whitespace length, not tokens.

4. **Recall evaluation** (`cal_recall`) zips gold-doc-id lists with
   `ret_res.values()`. JSON object key order must match gold list order (qid
   0,1,2,…). Do not reorder the retrieval dict.

5. **`get_gold_docs` for MuSiQue** (`paragraphs` branch) always builds
   `gold_doc_id` even if the first branch also set titles-only gold; the
   `contexts` branch never sets `gold_doc_id` (Hotpot/2Wiki use
   `supporting_facts`).

6. **ColBERT** is imported and unused. Ignore unless someone wires it.

---

## Evaluation

- Retrieval: Recall@{1,2,3,5} on gold passage **ids**.
- QA: token-normalized EM and F1 (SQuAD-style), max over aliases.

Paper also reports Recall@5 as the main retrieval metric and claims ~6× faster
online retrieval than HippoRAG 2 on MuSiQue from a smaller graph (~40% fewer
nodes). Replicate that comparison with wall-clock around `hg_diffusion` +
`struct_enhance`, excluding embedding encode time, and log `|E|`, `|D|`, nnz(H).

---

## Conventions

- Work from `HGRAG/`. `.env` is loaded as `Path(__file__).parents[2] / ".env"`
  from `inferencer.py`.
- NER JSON keys: query `entities`, corpus `named_entities`. Prompts must contain
  the word `json` (`response_format=json_object`).
- Do not commit `.env`, `output/`, `logs/`, or processed `data/samples/*` except
  `data/samples/raw/`.
- Prefer extending `Retriever` over rewriting `HG` / `GRAG` when swapping
  encoders. Diffusion math should stay encoder-agnostic.
