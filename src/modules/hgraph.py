import pandas as pd
import torch
from collections import defaultdict
from itertools import chain


class HG:
    def __init__(self, ent_did_path, device='cuda'):
        self.device = device
        df = pd.read_csv(ent_did_path, keep_default_na=False)
        labels, unique_ents = pd.factorize(df['ent'])
        df['eid'] = labels
        self.ent2id = dict(zip(unique_ents, range(len(unique_ents))))
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.ent_num = len(self.ent2id)
        self.doc_num = df['did'].max() + 1
        self.hg_df = df

        ent2doc = defaultdict(set)
        doc2ent = defaultdict(set)
        for eid, did in self.hg_df[['eid', 'did']].values:
            ent2doc[eid].add(did)
            doc2ent[did].add(eid)
        self.ent2doc, self.doc2ent = ent2doc, doc2ent

        self._create_hgraph()

    def _create_hgraph(self):
        indices = torch.stack([torch.tensor(self.hg_df['eid'].values), torch.tensor(self.hg_df['did'].values)], dim=0)
        values = torch.ones(indices.shape[1])
        H_ = torch.sparse_coo_tensor(indices, values, (self.ent_num, self.doc_num)).coalesce().to(self.device)
        self.H = torch.sparse_coo_tensor(H_.indices(), torch.ones_like(H_.values()), H_.size()).to(self.device)

        De = torch.sum(self.H, dim=0)
        Dv = torch.sum(self.H, dim=1)
        self.De_inv = torch.sparse_coo_tensor(indices=De.indices().repeat(2, 1),
                                              values=1.0 / De.values(),
                                              size=(self.doc_num, self.doc_num)).to(self.device)
        self.Dv_inv_sqrt = torch.sparse_coo_tensor(indices=Dv.indices().repeat(2, 1),
                                                   values=1.0 / torch.sqrt(Dv.values()),
                                                   size=(self.ent_num, self.ent_num)).to(self.device)
        if torch.isinf(self.De_inv).any():
            self.De_inv = torch.nan_to_num(self.De_inv, posinf=0.0, neginf=0.0)
        if torch.isinf(self.Dv_inv_sqrt).any():
            self.Dv_inv_sqrt = torch.nan_to_num(self.Dv_inv_sqrt, posinf=0.0, neginf=0.0)

        self.I = torch.sparse_coo_tensor(torch.arange(self.ent_num).unsqueeze(0).repeat(2, 1),
                                         torch.ones(self.ent_num), (self.ent_num, self.ent_num)).to(self.device)


    def save_ents(self, save_path):
        df = pd.DataFrame(self.id2ent.items(), columns=["Key", "Value"])
        df.to_csv(save_path, sep="\t", index=False, header=False, encoding="utf-8")


    def _get_Wp(self, doc_sims):
        index, values = [], []
        for did in range(self.doc_num):
            index.append(did)
            values.append(doc_sims[did])

        values = torch.tensor(values)
        W = torch.sparse_coo_tensor(indices=torch.tensor(index).repeat(2, 1),
                                    values=values,
                                    size=(self.doc_num, self.doc_num)).to(self.device)
        return W

    def _propagate(self, A, X, step=1):
        Xn = X.clone()
        for _ in range(step):
            Xn = A @ Xn
        return Xn

    def _sort_docs(self, P):
        if P.is_sparse:
            P = P.to_dense()
        sval, sidx = torch.sort(P.squeeze(), dim=0, descending=True)
        return sidx.tolist(), sval.tolist()

    def hg_diffusion(self, ent_ids, ent_sims, qd_sims, beta=0.5, step=1, multihot=True):
        X = torch.zeros(self.ent_num).scatter_(0, torch.tensor(ent_ids), 1).unsqueeze(1).to(self.device)
        if not multihot:
            X = torch.zeros(self.ent_num).scatter_reduce_(0, torch.tensor(ent_ids), torch.tensor(ent_sims), reduce="amax").unsqueeze(1).to(self.device)


        Wp = self._get_Wp(qd_sims)
        L_ = self.Dv_inv_sqrt @ self.H @ Wp @ self.De_inv @ self.H.T @ self.Dv_inv_sqrt

        P = (beta * Wp.coalesce().values().unsqueeze(dim=1) +
             (1 - beta) * (Wp @ (self.H.T @ self._propagate(L_, X, step))))

        dids_sorted, pvals_sorted = self._sort_docs(P)
        return dids_sorted, pvals_sorted

    def struct_enhance(self, ret_res, topk1=5, topk2=10):
        assert topk1 < topk2

        A = self.H.T @ self.H
        hadj = {}
        num_hyperedges = A.shape[0]

        for i in range(num_hyperedges):
            neighbors = torch.nonzero(A[i].to_dense() > 0, as_tuple=False).squeeze().tolist()
            if isinstance(neighbors, int):
                neighbors = [neighbors]
            hadj[i] = neighbors

        res = {}
        for qid, (docs, scores) in ret_res.items():
            sdocs = docs[:topk1]
            ndocs = set(chain.from_iterable([hadj[i] for i in sdocs]))
            fdocs, fscores = sdocs, scores[:topk1]
            for i, d in enumerate(docs[topk1: topk2]):
                if d in ndocs:
                    fdocs.append(d)
                    fscores.append(scores[topk1 + i])
            res[qid] = [fdocs, fscores]

        return res