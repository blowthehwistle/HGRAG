import argparse
from src.modules.grag import GRAG

def grag_retrieval(data_path, corpus_path, e2e_ret_path, q2d_ret_path, qe_path, grag_docs_path, hg_path=None, ent_did_path=None, beta=0.5, step=1, ent_thr=0.0, ent_topk=1, multihot=True, topk1=5, topk2=10, grag_ret_path=None, device='cuda'):
    grag = GRAG(data_path, corpus_path, hg_path, ent_did_path, device)

    ent_seeds = grag.get_seed(e2e_ret_path, q2d_ret_path, qe_path, ent_thr, ent_topk)
    grag.hg_retrieval(ent_seeds, grag_docs_path, beta=beta, step=step, topk1=topk1, topk2=topk2, multihot=multihot, ret_res_path=grag_ret_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper Graph RAG retrieval")

    parser.add_argument("--data_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--corpus_path", type=str, required=True, help="Corpus path")
    parser.add_argument("--e2e_ret_path", type=str, required=True, help="Ent(query)-to-ent(doc) retrieval result file")
    parser.add_argument("--q2d_ret_path", type=str, required=True, help="Query-to-doc retrieval result file")
    parser.add_argument("--qe_path", type=str, required=True, help="{Query: entid} file")
    parser.add_argument("--grag_docs_path", type=str, required=True, help="Output path of GRAG retrieved docs")

    parser.add_argument("--hg_path", type=str, default=None, help="Hyper graph dill file")
    parser.add_argument("--ent_did_path", type=str, default=None, help="Entity-to-docid mapping file")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")

    parser.add_argument("--ent_thr", type=float, default=0.0, help="Threshold for entity sim scores")
    parser.add_argument("--ent_topk", type=int, default=1, help="Top-k entity seeds")
    parser.add_argument("--no-multihot", dest="multihot",  action="store_false", help="Whether use multi-hot encoding X")
    parser.add_argument("--beta", type=float, default=0.5, help="Semantic enhancement coefficient")
    parser.add_argument("--step", type=int, default=1, help="Diffusion step")
    parser.add_argument("--topk1", type=int, default=5, help="Top-k docs")
    parser.add_argument("--topk2", type=int, default=10, help="If topk2!=0, perform structural enhancement over range [topk1, topk2]")

    parser.add_argument("--grag_ret_path", type=str, default=None, help="Output path of retrieval results with id")

    args = parser.parse_args()

    grag_retrieval(
        data_path=args.data_path,
        corpus_path=args.corpus_path,
        e2e_ret_path=args.e2e_ret_path,
        q2d_ret_path=args.q2d_ret_path,
        qe_path=args.qe_path,
        grag_docs_path=args.grag_docs_path,
        hg_path=args.hg_path,
        ent_did_path=args.ent_did_path,
        device=args.device,
        ent_thr=args.ent_thr,
        ent_topk=args.ent_topk,
        multihot=args.multihot,
        beta=args.beta,
        step=args.step,
        topk1=args.topk1,
        topk2=args.topk2,
        grag_ret_path=args.grag_ret_path
    )