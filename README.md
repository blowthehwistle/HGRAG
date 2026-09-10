# HGRAG

Code for the paper **Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop Question Answering**.

The benchmark datasets follow [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), and parts of the implementation and prompt templates are adapted from the HippoRAG repository.


## Project Structure

```text
src/
  data_processing.py      Data preprocessing and result postprocessing
  ent_extraction.py       Query/corpus entity extraction
  retrieval.py            Dense retrieval entrypoint
  build_hgraph.py         Hypergraph construction
  grag_retrieval.py       Hypergraph retrieval entrypoint
  qa.py                   QA generation
  prompts.py              Prompt templates
  utils.py                Shared utilities
  evaluation/             Retrieval and QA evaluation
  modules/
    retriever.py          Dense retrievers
    hgraph.py             Hypergraph and diffusion
    grag.py               HGRAG pipeline logic
    inferencer.py         LLM inference wrappers
    dataload.py           Prompted dataloaders
```


## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

### OpenAI setup 


1. `HGRAG/.env` (gitignored) 파일 만들고 다음과 같이 입력:

```bash
OPENAI_API_KEY=sk-... 
```

2. 모델 변경 (default below):

- `scripts/run_ent_extraction.sh` → `MODEL_ID=gpt-4o-mini`
- `scripts/run_qa.sh` → `MODEL_ID=gpt-4o-mini`

3. Quick smoke test for entity extraction:

```bash
mkdir -p /tmp/hgrag_smoke
python - <<'EOF'
import json
from pathlib import Path
Path("/tmp/hgrag_smoke/q.json").write_text(json.dumps([
    {"qid": 0, "question": "What city is the Eiffel Tower located in?"}
]), encoding="utf-8")
Path("/tmp/hgrag_smoke/c.json").write_text(json.dumps([
    {"did": 0, "title": "Paris",
     "text": "Paris is the capital of France. The Eiffel Tower is there."}
]), encoding="utf-8")
EOF

python -m src.ent_extraction \
  --model_id gpt-4o-mini \
  --data_path /tmp/hgrag_smoke/q.json \
  --resp_path /tmp/hgrag_smoke/q_ner.jsonl \
  --type Query \
  --max_batch_tokens 1000 \
  --max_new_tokens 200 \
  --log_path /tmp/hgrag_smoke/ner.log

python -m src.ent_extraction \
  --model_id gpt-4o-mini \
  --data_path /tmp/hgrag_smoke/c.json \
  --resp_path /tmp/hgrag_smoke/c_ner.jsonl \
  --type Corpus \
  --max_batch_tokens 1000 \
  --max_new_tokens 200 \
  --log_path /tmp/hgrag_smoke/ner.log

cat /tmp/hgrag_smoke/q_ner.jsonl   # expect key: entities
cat /tmp/hgrag_smoke/c_ner.jsonl   # expect key: named_entities
```

Notes:

- NER uses OpenAI `response_format=json_object`; prompts must mention the word `json`.
- QNER output key: `entities`. CNER output key: `named_entities`.
- Run commands from the `HGRAG/` directory so `.env` and `python -m src.*` resolve correctly.

### Sample / full pipeline

All run scripts take an optional dataset argument: `samples` (default) | `musique` | `hotpot` | `2wiki`.

```bash
DATASET=hotpot   # or samples / musique / 2wiki

bash scripts/run_data_processing.sh "$DATASET"
bash scripts/run_ent_extraction.sh "$DATASET"   # OpenAI NER; expensive on full corpora
bash scripts/run_build_hgraph.sh "$DATASET"
bash scripts/run_retrieval.sh "$DATASET"        # OpenAI text-embedding-3-small
bash scripts/run_grag_retrieval.sh "$DATASET"
bash scripts/run_qa.sh "$DATASET"               # OpenAI QA
bash scripts/run_evaluation.sh "$DATASET"
```

Outputs go under `data/{dataset}/`, `output/**/{dataset}/`, and `logs/`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wang2026cross,
  title={Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop Question Answering},
  author={Wang, Changjian and Deng, Weihong and Guan, Weili and Lu, Quan and Jiang, Ning},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={39},
  pages={33368--33376},
  year={2026}
}
```
