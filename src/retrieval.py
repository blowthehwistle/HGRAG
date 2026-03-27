from src.modules.retriever import Retriever, NVERetriever
import os
import argparse

def retrieval(model_path, corpus_path, query_path, corpus_vecs_path, query_vecs_path, retrival_res_path, k=100, query_chunk=10, corpus_chunk=4, reindex=False, device='cuda', max_token=100, prompt='query2passages', model_type='nve'):
    R = NVERetriever(model_path, device) if model_type == 'nve' else Retriever(model_path, device)
    if reindex:
        q = False if prompt is None else True
        print('Indexing query')
        R.indexing(query_path, query_vecs_path, query=q, max_token=max_token, prompt=prompt)
        print('Indexing corpus')
        R.indexing(corpus_path, corpus_vecs_path, max_token=max_token)
    elif not os.path.exists(corpus_vecs_path):
        R.indexing(corpus_path, corpus_vecs_path)
    elif not os.path.exists(query_vecs_path):
        R.indexing(query_path, query_vecs_path)

    R.retrieval(query_vecs_path, corpus_vecs_path, k=k, query_chunk=query_chunk, corpus_chunk=corpus_chunk,
                save_path=retrival_res_path, method='torch')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense Retrieval")

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--query_path", required=True)
    parser.add_argument("--corpus_vecs_path", required=True)
    parser.add_argument("--query_vecs_path", required=True)
    parser.add_argument("--retrival_res_path", required=True)

    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--query_chunk", type=int, default=10)
    parser.add_argument("--corpus_chunk", type=int, default=4)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_token", type=int, default=100)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model_type", default="nve", choices=["nve", "contriever"])

    args = parser.parse_args()

    retrieval(
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        query_path=args.query_path,
        corpus_vecs_path=args.corpus_vecs_path,
        query_vecs_path=args.query_vecs_path,
        retrival_res_path=args.retrival_res_path,
        k=args.k,
        query_chunk=args.query_chunk,
        corpus_chunk=args.corpus_chunk,
        reindex=args.reindex,
        device=args.device,
        max_token=args.max_token,
        prompt=args.prompt,
        model_type=args.model_type
    )