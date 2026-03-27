import json
import dill
from src.modules.hgraph import HG

class GRAG:
    def __init__(self, data_path, corpus_path, hg_path=None, ent_did_path=None, device='cuda'):
        if hg_path is not None:
            with open(hg_path, 'rb') as f:
                self.hg = dill.load(f)
        elif ent_did_path is not None:
            self.hg = HG(ent_did_path, device)
        else:
            raise ValueError("Either hg_path or ent_did_path must be provided.")
        self.data_path = data_path
        self.corpus_path = corpus_path

    def _get_e2e(self, e2e_ret_path, qe_path, ent_topk):
        with open(e2e_ret_path, 'r', encoding='utf-8') as f:
            e2e_data = json.load(f)

        with open(qe_path, 'r', encoding='utf-8') as f:
            qe_data = json.load(f)

        for q in qe_data:
            q['entities'] = {k: [e2e_data[str(k)][0][:ent_topk], e2e_data[str(k)][1][:ent_topk]] for k in q['entities']}

        return qe_data

    def get_seed(self, e2e_ret_path, q2d_ret_path, qe_path, ent_thr=0.0, ent_topk=1):
        data = self._get_e2e(e2e_ret_path, qe_path, ent_topk)
        qdsim = self._get_qdsim(q2d_ret_path)
        seeds = {}

        for q in data:
            s_ent, s_ent_v = [], []
            for ents, scores in q['entities'].values():
                es, vs = ents[:ent_topk], scores[:ent_topk]
                if ent_thr:
                    for i in range(1, ent_topk):
                        if vs[i] < ent_thr:
                            es, vs = es[:i], vs[:i]
                            break
                s_ent += es
                s_ent_v += vs
            seeds[str(q['qid'])] = (s_ent, s_ent_v, qdsim[q['qid']])

        return seeds

    def _get_qdsim(self, q2d_ret_file):
        with open(q2d_ret_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        qdsim = {}
        for qid, (doc_ids, scores) in data.items():
            qdsim[int(qid)] = dict(zip(doc_ids, scores))

        return qdsim

    def _get_docs(self, ret_res, save_path):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        id2doc = {}
        for d in corpus:
            id2doc[d['did']] = d['title'] + '\n' + d['text']

        id2query = {}
        for d in data:
            id2query[d['qid']] = d['question']

        res = {}
        for qid, (dids, scores) in ret_res.items():
            res[qid] = (id2query[int(qid)], [id2doc[did] for did in dids])

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)

    def hg_retrieval(self, seeds, save_path, beta=0.5, step=1, topk1=5, topk2=10, multihot=True, ret_res_path=None):
        ret_res = {}
        topk = topk2 if topk2 else topk1
        for qid, (eids, esim, dsim) in seeds.items():
            sidx, sval = self.hg.hg_diffusion(eids, esim, dsim, beta, step, multihot)
            ret_res[qid] = [sidx[:topk], sval[:topk]]
        if topk2:
            ret_res = self.hg.struct_enhance(ret_res, topk1, topk2)

        self._get_docs(ret_res, save_path)

        if ret_res_path:
            with open(ret_res_path, "w", encoding="utf-8") as f:
                json.dump(ret_res, f, ensure_ascii=False)



