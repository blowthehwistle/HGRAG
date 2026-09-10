import json
import pickle

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from colbert.data import Queries
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import faiss
import gc
from tqdm import tqdm

from src.prompts import PROMPTS
from src.utils import ensure_dir


from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).resolve().parents[2] / ".env")



class ColbRetriever:
    def __init__(self, checkpoint_path, work_path, experiment="colbertv2"):
        self.checkpoint_path = checkpoint_path
        self.root_path = work_path
        self.experiment = experiment

    def indexing(self, index_name, collection_path, nbits=2, index_bsize=64, overwrite=True):
        with Run().context(RunConfig(nranks=1, experiment=self.experiment, root=self.root_path)):
            config = ColBERTConfig(
                nbits=nbits,
                root=self.root_path,
                index_bsize=index_bsize,
            )
            indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            indexer.index(name=index_name, collection=collection_path, overwrite=overwrite)

    def retrieval(self, index_name, query_path, k=5, norm=False, save_path=None):
        with Run().context(RunConfig(nranks=1, experiment=self.experiment, root=self.root_path)):
            config = ColBERTConfig(
                root=self.root_path,
            )
            searcher = Searcher(index=index_name, config=config)
            queries = Queries(query_path)
            ranking = searcher.search_all(queries, k=k)
            ranking_dict = {}
            for qid, rank in ranking.data.items():
                max_score = rank[0][2] if norm else 1.0
                ranking_dict[qid] = ([r[0] for r in rank], [r[2] / max_score for r in rank])
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(ranking_dict, f, ensure_ascii=False)
            return ranking_dict




class Retriever:
    def __init__(self, checkpoint_path, device='cuda'):
        self.plm = AutoModel.from_pretrained(checkpoint_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.device = device


    def indexing(self, data_path, save_path, query=False, dsort=True, max_token=5000, max_length=None, prompt='query2passages'):
        cdf = pd.read_csv(data_path, sep='\t', header=None, na_filter=False, dtype={1: 'string'})
        original_indices = cdf.index
        if dsort:
            sorted_indices = cdf[1].apply(len).sort_values().index
            cdf_sorted = cdf.iloc[sorted_indices]
            cdf = cdf_sorted.reset_index(drop=True)
            original_indices = sorted_indices.argsort()
        strs = cdf[1].tolist()
        vecs = self._encode(strs, max_token, query, max_length, prompt=prompt)
        if dsort:
            vecs = vecs[original_indices]
        dsv = {'s2v': {s: v for s, v in zip(strs, vecs)}, 'vecs': vecs}

        ensure_dir(save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(dsv, f)
        return dsv

    def _encode(self, strs, max_token, query=False, max_length=None, pool='mean', prompt=None):
        all_vecs = []
        for ss in self._split_strs(strs, max_token):
            inputs = self.tokenizer(ss, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.plm(**inputs).last_hidden_state

            if pool == 'mean':
                mask = inputs["attention_mask"]
                token_embeddings = outputs.masked_fill(~mask[..., None].bool(), 0.)
                pooled_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            else:
                pooled_embeddings = outputs
            all_vecs.append(pooled_embeddings.cpu().numpy())

        all_vecs = np.vstack(all_vecs)
        return all_vecs


    def _split_strs(self, str_list, max_token, token_func='tokenizer'):
        current_list = []
        current_length = 0
        for string in str_list:
            string_length = len(self.tokenizer.tokenize(string)) if token_func == 'tokenizer' else len(str(string).split())
            if current_length + string_length <= max_token:
                current_list.append(string)
                current_length += string_length
            else:
                if current_list:
                    yield current_list
                    current_list = [string]
                    current_length = string_length
                else:
                    yield [string]
        if current_list:
            yield current_list

    def retrieval_faiss(self, query_vecs_path, corpus_vecs_path, k=2047, query_chunk=10, corpus_chunk=4, save_path=None):
        with open(corpus_vecs_path, 'rb') as f:
            corps = pickle.load(f)
            original_vecs = corps['vecs']

        with open(query_vecs_path, 'rb') as f:
            query = pickle.load(f)
            new_vecs = query['vecs']

        if len(original_vecs) == 0 or len(new_vecs) == 0:
            return {}

        original_vecs = original_vecs.astype(np.float32)
        new_vecs = new_vecs.astype(np.float32)

        faiss.normalize_L2(original_vecs)
        faiss.normalize_L2(new_vecs)

        # Preparing Data for k-NN Algorithm
        print('Chunking')

        dim = len(original_vecs[0])
        index_split = 4  # 4
        index_chunks = np.array_split(original_vecs, corpus_chunk)
        query_chunks = np.array_split(new_vecs, query_chunk)

        # Building and Querying FAISS Index by parts to keep memory usage manageable.
        print('Building Index')

        index_chunk_D = []
        index_chunk_I = []

        current_zero_index = 0

        for num, index_chunk in enumerate(index_chunks):

            print('Running Index Part {}'.format(num))
            index = faiss.IndexFlat(dim,
                                    faiss.METRIC_INNER_PRODUCT)

            if faiss.get_num_gpus() > 1:
                gpu_resources = []

                for i in range(faiss.get_num_gpus()):
                    res = faiss.StandardGpuResources()
                    gpu_resources.append(res)

                gpu_index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index)
            else:
                gpu_resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

            print()
            gpu_index.add(index_chunk)

            D, I = [], []

            for q in tqdm(query_chunks):
                d, i = gpu_index.search(q, k)
                i += current_zero_index

                D.append(d)
                I.append(i)

            index_chunk_D.append(D)
            index_chunk_I.append(I)

            current_zero_index += len(index_chunk)

            del gpu_index
            del gpu_resources
            gc.collect()

        print('Combining Index Chunks')

        stacked_D = []
        stacked_I = []

        for D, I in zip(index_chunk_D, index_chunk_I):
            D = np.vstack(D)
            I = np.vstack(I)

            stacked_D.append(D)
            stacked_I.append(I)

        del index_chunk_D
        del index_chunk_I
        gc.collect()

        stacked_D = np.hstack(stacked_D)  ### cnum*qnum*k -> qnum*(cnum*k)
        stacked_I = np.hstack(stacked_I)

        full_sort_I = []
        full_sort_D = []

        for d, i in tqdm(zip(stacked_D, stacked_I)):
            sort_indices = np.argsort(d, kind='stable')

            sort_indices = sort_indices[::-1]

            i = i[sort_indices][:k]
            d = d[sort_indices][:k]

            full_sort_I.append(i)
            full_sort_D.append(d)

        del stacked_D
        del stacked_I
        gc.collect()

        sorted_candidate_dictionary = {}

        for qi, (di, s) in enumerate(zip(full_sort_I, full_sort_D)):
            sorted_candidate_dictionary[qi] = (di.tolist(), s.tolist())

        if save_path:
            ensure_dir(save_path)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_candidate_dictionary, f, ensure_ascii=False)

        return sorted_candidate_dictionary  ### {'qid1':([did1, did2,...],[score1, score2,...])}

    def retrieval_torch(self, query_vecs_path, corpus_vecs_path, k=2047, query_chunk=10, corpus_chunk=4, save_path=None,
                        device='cuda'):
        with open(corpus_vecs_path, 'rb') as f:
            corps = pickle.load(f)
            corpus_vecs = corps['vecs']

        with open(query_vecs_path, 'rb') as f:
            query = pickle.load(f)
            query_vecs = query['vecs']

        corpus_vecs = torch.from_numpy(corpus_vecs).to(device)
        query_vecs = torch.from_numpy(query_vecs).to(device)
        corpus_vecs = F.normalize(corpus_vecs, p=2, dim=1)
        query_vecs = F.normalize(query_vecs, p=2, dim=1)

        print('Chunking')
        index_chunks = torch.chunk(corpus_vecs, corpus_chunk)
        query_chunks = torch.chunk(query_vecs, query_chunk)

        index_chunk_D = []
        index_chunk_I = []
        current_zero_index = 0

        print('Building Index')
        for num, index_chunk in enumerate(index_chunks):
            print(f'Running Index Part {num}')

            D, I = [], []
            for q in tqdm(query_chunks):
                similarity = torch.matmul(q, index_chunk.T)  # (q_chunk_size, index_chunk_size)
                d, i = similarity.topk(min(k, similarity.shape[1]), dim=1, largest=True, sorted=True)

                i += current_zero_index
                D.append(d.cpu().numpy())
                I.append(i.cpu().numpy())

            index_chunk_D.append(D)
            index_chunk_I.append(I)

            current_zero_index += index_chunk.shape[0]

            gc.collect()

        print('Combining Index Chunks')
        stacked_D = np.hstack([np.vstack(D) for D in index_chunk_D])
        stacked_I = np.hstack([np.vstack(I) for I in index_chunk_I])

        full_sort_I = []
        full_sort_D = []

        for d, i in tqdm(zip(stacked_D, stacked_I)):
            sort_indices = np.argsort(d)[::-1]
            i = i[sort_indices][:k]
            d = d[sort_indices][:k]
            full_sort_I.append(i)
            full_sort_D.append(d)

        del stacked_D, stacked_I
        gc.collect()

        sorted_candidate_dictionary = {qi: (di.tolist(), s.tolist()) for qi, (di, s) in
                                       enumerate(zip(full_sort_I, full_sort_D))}

        if save_path:
            ensure_dir(save_path)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_candidate_dictionary, f, ensure_ascii=False)

        return sorted_candidate_dictionary ### {'qid1':([did1, did2,...],[score1, score2,...])}

    def retrieval(self, query_vecs_path, corpus_vecs_path, k=2047, query_chunk=10, corpus_chunk=4, save_path=None,
                  device='cuda', method='torch'):
        if method == 'torch':
            return self.retrieval_torch(query_vecs_path, corpus_vecs_path, k, query_chunk, corpus_chunk, save_path,
                                        device)
        elif method == 'faiss':
            return self.retrieval_faiss(query_vecs_path, corpus_vecs_path, k, query_chunk, corpus_chunk, save_path)


    def __del__(self):
        if hasattr(self, "plm"):
            del self.plm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class NVERetriever(Retriever):
    def __init__(self, checkpoint_path, device='cuda'):
        self.plm = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, device_map="auto")
        self.device = device

    def _encode(self, strs, max_token, query=False, max_length=None, pool='mean', prompt='query2passages'):
        all_vecs = []
        max_length = max_length if max_length else 32768
        prefix = "Instruct: " + PROMPTS["NVEmd"][prompt] + "\nQuery: " if query else ""
        for ss in self._split_strs(strs, max_token, 'None'):
            print('indexing---', len(ss))
            outputs = self.plm.encode(ss, instruction=prefix, max_length=max_length)
            all_vecs.append(outputs.cpu().numpy())

        all_vecs = np.vstack(all_vecs)
        return all_vecs

class OpenAIEmbedRetriever(Retriever):
    
    def __init__(self, model_id="text-embedding-3-small", device="cpu", batch_size=64):
        # GPU/로컬 모델 로드 안 함
        self.model_id = model_id
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.batch_size = batch_size
        self.device = device  # 안 써도 OK, 시그니처 맞춤용

    def _encode(self, strs, max_token, query=False, max_length=None, pool="mean", prompt=None):
        # query/prompt는 OpenAI에선 무시해도 됨
        all_vecs = []
        for i in range(0, len(strs), self.batch_size):
            batch = strs[i:i + self.batch_size]
            # 빈 문자열 방지
            batch = [s if s.strip() else " " for s in batch]
            resp = self.client.embeddings.create(model=self.model_id, input=batch)
            # resp.data 순서가 input과 같다고 보장됨
            vecs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
            all_vecs.append(np.asarray(vecs, dtype=np.float32))
        return np.vstack(all_vecs)
