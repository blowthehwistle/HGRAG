from src.modules.hgraph import HG
import dill
import argparse

def build_graph(ent_did_path, hg_save_path=None, ent_id_save_path=None, device='cuda'):
    G = HG(ent_did_path, device=device)
    if hg_save_path:
        with open(hg_save_path, "wb") as f:
            dill.dump(G, f)
    if ent_id_save_path:
        G.save_ents(ent_id_save_path)

def main():
    parser = argparse.ArgumentParser(description="Build hypergraph")

    parser.add_argument("--ent_did_path", type=str, required=True,
                        help="Path to entity-document mapping file")
    parser.add_argument("--hg_save_path", type=str, default=None,
                        help="Path to save serialized hypergraph")
    parser.add_argument("--ent_id_save_path", type=str, default=None,
                        help="Path to save entity id mapping tsv file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    build_graph(
        ent_did_path=args.ent_did_path,
        hg_save_path=args.hg_save_path,
        ent_id_save_path=args.ent_id_save_path,
        device=args.device
    )


if __name__ == "__main__":
    main()