from src.utils import extract_json_dict, setup_logger, ensure_dir
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# api key load from .env file (src/modules/inferencer.py → parents[2] = HGRAG/)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class Inferencer:
    # json mode : openai model에서 json 형식으로 안정적 출력 가능 (true : NER, false : QA)
    def __init__(self, model_id, resp_path, dataloader, max_new_tokens=1000, device='auto', temperature=None, log_path='inference.log', json_mode=False):
        self.logger = setup_logger(__name__, log_path)
        self.logger.info("Init OpenAI client...")
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = 0 if temperature is None else temperature
        self.json_mode = json_mode

        self.dataloader = dataloader
        self.resp_path = resp_path
        self.logger.info(f"Loading data from {self.dataloader.datapath}")



    def infer(self):
        self.logger.info("Start infering...")

        ensure_dir(self.resp_path)
        with open(self.resp_path, "a", encoding="utf-8") as f:
            for i, bdata in enumerate(self.dataloader):
                try:
                    self.logger.info(f"Processing batch {i}, size {len(bdata)}...")
                    data = self._preprocess(bdata)
                    texts = [self._generate_one(messages) for messages in data]
                    out = [[{"generated_text": t}] for t in texts]
                    res = self._postprocess(bdata, out)
                    for item in res:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                except Exception as e:
                    self.logger.error(f"Error processing batch {i}: {e}; {bdata[0]}")
                    continue

        self.logger.info(f"Finished processing. Results saved to {self.resp_path}.")

    def _preprocess(self, bdata):
        return [item[1] for item in bdata]

    def _postprocess(self, rawdata, outdata):
        results = []
        for rd, out in zip(rawdata, outdata):
            results.append({
                "idd": rd[0],
                "extracted_data": extract_json_dict(
                    out[0]["generated_text"]
                ),
                "rawout": out
            })

        return results

    def _generate_one(self, messages):
        kwargs = dict(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""



class QAInferencer(Inferencer):
    def _postprocess(self, rawdata, outdata):
        results = []
        for rd, out in zip(rawdata, outdata):
            gtext = out[0]['generated_text'].split('Answer: ')
            ans = {'Answer': gtext[1], 'Thought': gtext[0]} if len(gtext) == 2 else {}
            results.append({
                "qid": rd[0],
                "extracted_data": ans,
                "rawout": out
            })
        return results
