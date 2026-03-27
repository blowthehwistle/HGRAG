import argparse
from src.modules.inferencer import Inferencer
from src.modules.dataload import QueryDataLoader, CorpusDataLoader

def extract(model_id, data_path, resp_path, max_batch_tokens, type, max_new_tokens=1000, temperature=0, device='auto', log_path='../logs/ent_extract.log'):
    if type == 'Query':
        dataloader = QueryDataLoader(data_path, max_batch_tokens, prompt_key='QNER')
    elif type == 'Corpus':
        dataloader = CorpusDataLoader(data_path, max_batch_tokens, prompt_key='CNER')

    inferencer = Inferencer(model_id, resp_path, dataloader, max_new_tokens=max_new_tokens, device=device, temperature=temperature, log_path=log_path)
    inferencer.infer()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run entity extraction for Query or Corpus data")

    parser.add_argument("--model_id", type=str, required=True, help="Path or name of the model")
    parser.add_argument("--data_path", type=str, required=True, help="Input data file path")
    parser.add_argument("--resp_path", type=str, required=True, help="Output response file path")
    parser.add_argument("--type", type=str, choices=["Query", "Corpus"], required=True, help="Data type: Query or Corpus")
    parser.add_argument("--max_batch_tokens", type=int, default=1000, help="Max tokens per batch")
    parser.add_argument("--max_new_tokens", type=int,default=1000, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto", help="Device to use, e.g. auto / cuda / cpu")
    parser.add_argument("--log_path", type=str, default="../logs/ent_extract.log", help="Log file path")

    args = parser.parse_args()

    extract(
        model_id=args.model_id,
        data_path=args.data_path,
        resp_path=args.resp_path,
        max_batch_tokens=args.max_batch_tokens,
        type=args.type,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        log_path=args.log_path
    )