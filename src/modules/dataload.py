import json
from src.prompts import add_prompt


class TokenBatchDataLoader:
    def __init__(self, data_path, max_tokens, prompt_key=None):
        self.max_tokens = max_tokens
        self.prompt_key = prompt_key
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.text = self._process_data(self.data)
        self.batches = self._create_batches()
        self.datapath = data_path

    def _create_batches(self):
        batches = []
        batch = []
        batch_token_count = 0
        for sentence in self.text:
            token_count = self._count_tokens(sentence)
            if batch and batch_token_count + token_count > self.max_tokens:
                batches.append(batch)
                batch = []
                batch_token_count = 0
            batch.append(sentence)
            batch_token_count += token_count
        if batch:
            batches.append(batch)

        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def _count_tokens(self, sentence):
        return len(str(sentence).split())

    def _process_data(self, data):
        return data


class QueryDataLoader(TokenBatchDataLoader):
    def _process_data(self, data):
        text = []
        for item in data:
            q = item['question']
            prompt = add_prompt(self.prompt_key, q)
            text.append((item['qid'], prompt))
        return text


class CorpusDataLoader(TokenBatchDataLoader):
    def _process_data(self, data):
        text = []
        for item in data:
            q = item['title'] + '\n' + item['text']
            prompt = add_prompt(self.prompt_key, q)
            text.append((item['did'], prompt))
        return text


class RetDocsDataLoader(TokenBatchDataLoader):
    def _process_data(self, data):
        text = []
        for qid, (q, docs) in data.items():
            s = ''
            for i, doc in enumerate(docs):
                s += 'Wikipedia Title: ' + doc + '\n'
            s += '\nQuestion: ' + q + '\nThought: '
            prompt = add_prompt(self.prompt_key, s)
            text.append((qid, prompt))
        return text