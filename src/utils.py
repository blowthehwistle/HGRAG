import re
import json
import logging
import os



def setup_logger(name: str, log_file: str = 'app.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def extract_json_dict(text):
    pattern = r'\{.*?\}'
    matches = re.findall(pattern, text, re.S)

    result = []
    for match in matches:
        try:
            json_dict = json.loads(match)
            result.append(json_dict)
        except json.JSONDecodeError:
            return []

    return result


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ensure_dir(path):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
