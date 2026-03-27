import json
import pandas as pd
import argparse
from src.utils import read_jsonl, ensure_dir

def get_gold_docs(data_path, dataset_name=None, save_path=None):
    samples = json.load(open(data_path, "r"))
    gold_docs, gold_docs_id = [], []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
            gold_doc_id = set([item[-1] for item in sample['supporting_facts']])
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]
            gold_doc_id = [item['did'] for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)

        gold_doc_id = list(set(gold_doc_id))
        gold_docs_id.append(gold_doc_id)


    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(gold_docs, f, ensure_ascii=False, indent=4)
        with open(save_path.replace(".json", "_id.json"), 'w', encoding='utf-8') as f:
            json.dump(gold_docs_id, f, ensure_ascii=False, indent=4)

    return gold_docs, gold_docs_id


def get_gold_answers(data_path, save_path=None):
    samples = json.load(open(data_path, "r"))
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(list(gold_ans))

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(gold_answers, f, ensure_ascii=False, indent=4)

    return gold_answers


def corpus_add_ids(doc_path, save_path):
    with open(doc_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for index, item in enumerate(data):
        item['did'] = index
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def data_add_ids(data_path, corpus_id_path, save_path, dataname='musique'):
    with open(data_path, 'r', encoding='utf-8') as f1:
        file1_data = json.load(f1)
    with open(corpus_id_path, 'r', encoding='utf-8') as f2:
        file2_data = json.load(f2)

    if dataname == 'musique':
        text_to_did = {entry['text']: entry['did'] for entry in file2_data}

        for qid, item in enumerate(file1_data):
            item['qid'] = qid
            for paragraph in item['paragraphs']:
                text = paragraph['paragraph_text']
                if text in text_to_did:
                    paragraph['did'] = text_to_did[text]
                else:
                    paragraph['did'] = None
                    print(f"No match found for text: {text}")
    else:
        title_to_id = {entry['title']: entry['did'] for entry in file2_data}

        for qid, item in enumerate(file1_data):
            item['qid'] = qid
            for facts in item['supporting_facts']:
                if facts[0] in title_to_id:
                    facts.append(title_to_id[facts[0]])
                else:
                    print(f"No match found for title: {facts[0]}")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(file1_data, f, ensure_ascii=False, indent=4)


def qner_res_process(data_path, save_path, save_qe_path):
    data = read_jsonl(data_path)
    res, kw_ents = [], set()
    for item in data:
        kws = item['extracted_data']
        ents = []
        for kw in kws:
            for e in kw['entities']:
                if e not in ents:
                    ents.append(e)
        res.append({"qid": item['idd'], "entities": ents})
        kw_ents.update(ents)

    kw_ents = sorted(kw_ents)
    qe2id = {kw: id for id, kw in enumerate(kw_ents)}
    ensure_dir(save_path)
    with open(save_path, 'w', encoding='utf-8') as f:
        for id, p in enumerate(kw_ents):
            f.write(f"{id}\t\"{p}\"" + '\n')

    ensure_dir(save_qe_path)
    with open(save_qe_path, 'w', encoding='utf-8') as f:
        f.write("[\n")
        for i, r in enumerate(res):
            r['entities'] = [qe2id[kw] for kw in r['entities']]
            f.write(json.dumps(r, ensure_ascii=False) + (",\n" if i != len(res) - 1 else "\n"))
        f.write("]")


def cner_res_process(data_path, save_path):
    data = read_jsonl(data_path)
    dent = {}
    for x in data:
        id = x['idd']
        try:
            ents = x['extracted_data'][0]['named_entities']
        except Exception:
            print('Extract failed', id)
            ents = []
        dent[id] = ents

    rows = [(v, k) for k, values in dent.items() for v in values]
    df = pd.DataFrame(rows, columns=["ent", "did"])
    ensure_dir(save_path)
    df.to_csv(save_path, index=False, encoding='utf-8')


def create_query_tsv(data_path, save_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    querys = [item['question'].replace('"', '""') for item in corpus]

    ensure_dir(save_path)
    with open(save_path, 'w') as f:
        for pid, p in enumerate(querys):
            f.write(f"{pid}\t\"{p}\"" + '\n')


def create_doc_tsv(data_path, save_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    corpus_contents = [(item['title'] + '\n' + item['text']).replace('"', '""') for item in corpus]

    ensure_dir(save_path)
    with open(save_path, 'w') as f:
        for pid, p in enumerate(corpus_contents):
            f.write(f"{pid}\t\"{p}\"" + '\n')



def qa_res_process(data_path, save_path):
    data = read_jsonl(data_path)
    res = []
    for d in data:
        ans = d['extracted_data'].get('Answer')
        if ans:
            res.append(ans[:-1] if ans[-1]=='.' else ans)
        else:
            res.append('')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True,
                        choices=[
                            "get_gold_docs",
                            "get_gold_answers",
                            "corpus_add_ids",
                            "data_add_ids",
                            "qner_res_process",
                            "cner_res_process",
                            "qa_res_process",
                            "create_query_tsv",
                            "create_doc_tsv"
                        ])

    parser.add_argument("--data_path")
    parser.add_argument("--dataset_name")
    parser.add_argument("--save_path")
    parser.add_argument("--save_qe_path")
    parser.add_argument("--doc_path")
    parser.add_argument("--corpus_id_path")
    parser.add_argument("--dataname", default="musique")

    args = parser.parse_args()

    if args.task == "get_gold_docs":
        get_gold_docs(args.data_path, args.dataset_name, args.save_path)

    elif args.task == "get_gold_answers":
        get_gold_answers(args.data_path, args.save_path)

    elif args.task == "corpus_add_ids":
        corpus_add_ids(args.doc_path, args.save_path)

    elif args.task == "data_add_ids":
        data_add_ids(args.data_path, args.corpus_id_path, args.save_path, args.dataname)

    elif args.task == "qner_res_process":
        qner_res_process(args.data_path, args.save_path, args.save_qe_path)

    elif args.task == "cner_res_process":
        cner_res_process(args.data_path, args.save_path)

    elif args.task == "qa_res_process":
        qa_res_process(args.data_path, args.save_path)

    elif args.task == "create_query_tsv":
        create_query_tsv(args.data_path, args.save_path)

    elif args.task == "create_doc_tsv":
        create_doc_tsv(args.data_path, args.save_path)


if __name__ == "__main__":
    main()