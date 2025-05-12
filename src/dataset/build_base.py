import os, sys

sys.path.append(os.path.dirname(os.getcwd()))

import argparse
from collections import defaultdict
from tqdm import tqdm

import numpy as np

from utils import read_json_by_line, save_json


def build_base(args):
    jsonl_path = lambda name, types: (
        f"{name}.jsonl" if types == "data" else f"meta_{name}.jsonl"
    )
    data_path = os.path.join(args.folder, args.name, jsonl_path(args.name, "data"))
    meta_path = os.path.join(args.folder, args.name, jsonl_path(args.name, "meta"))

    history_dict = defaultdict(list)
    itemidx2meta = dict()
    item2idx, idx2item = dict(), dict()
    user2idx, idx2user = dict(), dict()
    uidx, iidx = 0, 0

    for line in read_json_by_line(meta_path, "Meta Data parsing"):
        item_id = line["parent_asin"]
        if item_id not in item2idx.keys():
            item2idx.update({item_id: iidx})
            idx2item.update({iidx: item_id})
            iidx += 1

        if item_id not in itemidx2meta.keys():
            idx = item2idx[item_id]
            itemidx2meta.update({idx: line})

    for line in read_json_by_line(data_path, "Review Data parsing"):
        user_id = line["user_id"]
        iidx = item2idx[line["parent_asin"]]
        timestamp = line["timestamp"]

        # history
        if user_id not in user2idx.keys():
            user2idx.update({user_id: uidx})
            idx2user.update({uidx: user_id})
            uidx += 1

        history_dict[user2idx[user_id]].append((iidx, timestamp))

    final_history = dict()

    for uidx, item_time in tqdm(history_dict.items(), desc="Sorting items"):
        items = np.array(sorted(item_time, key=lambda x: x[1]))[:, 0].tolist()
        final_history.update({uidx: items})

    for data, fname in tqdm(
        [
            (final_history, "history"),
            (itemidx2meta, "itemidx2meta"),
            (idx2user, "idx2user"),
            (idx2item, "idx2item"),
        ]
    ):
        save_json(data, args, fname)

    print(f"Finished saving base files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="/home/doyooni303/experiments/sequential_recsys/datasets/amazon",
        help="Dataset folder",
    )
    parser.add_argument("--name", type=str, help="Name of the dataset")
    args = parser.parse_args()
    build_base(args)
