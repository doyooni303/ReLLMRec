import os
import json
from tqdm import tqdm
from typing import List
import logging
import datetime
import os
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def read_json_by_line(path, desc: str):
    with open(path, "r") as f:
        for line in tqdm(f.readlines(), desc=desc):
            yield json.loads(line.rstrip())


def open_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]


def save_jsonl(data: list, path: str, desc) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in tqdm(data, desc=desc):
            json.dump(line, f, ensure_ascii=False)
            f.writelines("\n")


def save_json(data, args, fname: str):
    save_path = os.path.join(args.folder, args.name, f"{args.name}_{fname}.json")
    json.dump(data, open(save_path, "w"))


def write_text_file(file_path: str, data: List):
    with open(file_path, "w") as f:
        f.writelines(data)


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


class Log(object):
    def __init__(self, config):
        if config["train_mode"] == "epochs":
            name = f"{config['fname']}-top_k-{config['top_k']}-epochs-{config['max_epochs']}-{get_local_time()}"
        elif config["train_mode"] == "steps":
            name = f"{config['fname']}-top_k-{config['top_k']}-steps-{config['max_steps']}-{get_local_time()}"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level=logging.INFO)

        handler = logging.FileHandler(os.path.join(config["save_path"], name + ".log"))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add(self, text):
        self.logger.info(text)
