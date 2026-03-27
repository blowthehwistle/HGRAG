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

The repository includes sample data under `data/samples/raw/`. Run the sample pipeline in the following order:

```bash
bash scripts/run_data_processing.sh
bash scripts/run_ent_extraction.sh
bash scripts/run_build_hgraph.sh
bash scripts/run_retrieval.sh
bash scripts/run_grag_retrieval.sh
bash scripts/run_qa.sh
bash scripts/run_evaluation.sh
```

This produces intermediate files under `data/samples/`, `output/`, and `logs/`.

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
