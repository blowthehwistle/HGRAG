import argparse
from src.modules.inferencer import QAInferencer
from src.modules.dataload import RetDocsDataLoader

def qa(model_id, data_path, resp_path, max_batch_tokens=1000, prompt_key='QA', max_new_tokens=1000, temperature=0, device='auto', log_path='../logs/qa.log'):
    dataloader = RetDocsDataLoader(data_path, max_batch_tokens, prompt_key=prompt_key)
    inferencer = QAInferencer(model_id, resp_path, dataloader, max_new_tokens=max_new_tokens, device=device, temperature=temperature, log_path=log_path)
    inferencer.infer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA Inference")

    parser.add_argument("--model_id", type=str, required=True, help="Model ID or path")
    parser.add_argument("--data_path", type=str, required=True, help="GRAG_DOCS_PATH, hg retrieved docs")
    parser.add_argument("--resp_path", type=str, required=True, help="Output response path")
    parser.add_argument("--max_batch_tokens", default=1000, type=int, required=True, help="Max tokens per batch")
    parser.add_argument("--prompt_key", type=str, default="QA", help="Prompt key")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--log_path", type=str, default="../logs/qa.log", help="Log file path")

    args = parser.parse_args()

    qa(
        model_id=args.model_id,
        data_path=args.data_path,
        resp_path=args.resp_path,
        max_batch_tokens=args.max_batch_tokens,
        prompt_key=args.prompt_key,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        log_path=args.log_path
    )