from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.utils import extract_json_dict, setup_logger, ensure_dir
import json
import torch
import gc
import os

class Inferencer:
    def __init__(self, model_id, resp_path, dataloader, max_new_tokens=1000, device='auto', temperature=None, log_path='inference.log'):
        self.logger = setup_logger(__name__, log_path)
        self.logger.info("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype="auto", padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device)

        pipe_kwargs = dict(
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
        )

        if temperature is not None:
            pipe_kwargs.update(
                do_sample=temperature > 0,
                temperature=temperature
            )

        self.pipe = pipeline("text-generation", **pipe_kwargs)

        self.logger.info("Loading data...")
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
                    out = self.pipe(data, batch_size=len(data), return_full_text=False)
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

    def __del__(self):
        del self.model, self.tokenizer, self.pipe, self.dataloader, self.logger
        gc.collect()
        torch.cuda.empty_cache()


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