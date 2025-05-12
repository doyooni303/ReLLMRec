import os
import json
import random
from typing import Dict, Tuple

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from functools import lru_cache

from .utils import data_partition


class AmazonDataset(Dataset):
    def __init__(self, config, flag: str = None) -> None:
        self.config = config
        self.flag = flag
        self._set_parameters(self.config)
        self._load_data()
        # Cache for tokenized queries to avoid redundant processing
        self.cache = {}

    def _set_parameters(self, config):
        self.path = config["path"]
        self.fname = config["fname"]
        self.max_items = config["max_items"]
        self.min_items = config["min_items"]
        # Load tokenizer only once and reuse
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], use_fast=True  # Use the fast tokenizer
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_data(self):
        # Cache JSON files after loading
        self._json_cache = {}

        def _load_json(path: str, name: str) -> Dict:
            cache_key = f"{path}_{name}"
            if cache_key not in self._json_cache:
                self._json_cache[cache_key] = json.load(
                    open(os.path.join(path, name), "r")
                )
            return self._json_cache[cache_key]

        def _get_data(fname: str, path: str, min_items: int, flag: str):
            train, valid, test = data_partition(fname, path, min_items)
            self.train = train.copy()

            if flag == "train":
                data = train
            else:
                data = {k: list(v) for k, v in train.items()}  # Create a copy
                for user_id in data.keys():
                    if flag == "valid":
                        data[user_id].extend(valid[user_id])
                        data[user_id] = data[user_id][-(min_items + 1) :]
                    elif flag == "test":
                        # Fix the extend operation to properly append items
                        data[user_id].extend(valid[user_id])
                        data[user_id].extend(test[user_id])
                        data[user_id] = data[user_id][-(min_items + 1) :]
                
                # number of validation reduced
                if flag == 'valid':
                    random.seed(self.config['seed'])
                    valid_num = int(len(data) * 0.1)
                    valid_users = random.sample(list(data.keys()), valid_num)
                    data = {user: data[user] for user in valid_users}

            return data

        self.meta_name_dict = _load_json(self.path, f"{self.fname}_meta_name_dict.json")
        self.similar_users = _load_json(self.path, f"{self.fname}_similar_users.json")
        self.data = _get_data(self.fname, self.path, self.min_items, self.flag)
        self.usermap = {i: user_id for i, user_id in enumerate(self.data.keys())}

        # Precompute item metadata for common items
        self._item_metadata_cache = {}

    # Use LRU cache to avoid repeated metadata lookups
    @lru_cache(maxsize=1024)
    def _get_item_metadata(self, item_id: int) -> Dict:
        """Get metadata of the item with caching"""
        return {
            key: self.meta_name_dict[key][str(item_id)]
            for key in self.meta_name_dict.keys()
        }

    def _get_item_text(self, item_id: int, metadata: dict) -> str:
        item_text = f"<Item {item_id}>\n"
        item_text += "".join([f"{key}: {value}\n" for key, value in metadata.items()])
        return item_text[: self.config["max_length"]]

    def _format_item_list_query(
        self, user_id: str, data: dict, meta_name_dict: dict
    ) -> str:
        """Get the item list query for the user"""
        # Check if this query is already cached
        cache_key = f"item_list_{user_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = f"Here is a item list of <user {user_id}> that shows the preference of <user {user_id}> in a time-order, so the last item is the most recent item.\n"
        query += "[Item List]\n"

        history_items = data[user_id][-(self.max_items + 1) : -1]
        item_texts = []

        for i, item_id in enumerate(history_items):
            metadata = self._get_item_metadata(item_id)
            item_texts.append(f"{i}.\n{self._get_item_text(item_id, metadata)}")

        query += "\n".join(item_texts)
        query += f"\nPlease select the most related items based on the <user {user_id}>'s history item list."

        # Cache the result
        self.cache[cache_key] = query
        return query

    def _format_similar_users_query(
        self, user_id: str, data: dict, meta_name_dict: dict, similar_users: dict
    ) -> str:
        """Get the similar users query for the user"""
        # Check if this query is already cached
        cache_key = f"similar_users_{user_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = f"Here is a list of similar users of <user {user_id}> and their history item list. Items are represented with a title and the first user is the most similar user to <user {user_id}>.\n"
        query += "[Similar Users]\n"

        # Fix typo in original code: <useer should be <user
        similar_user_list = [l[0] for l in similar_users[str(user_id)]]
        user_texts = []

        for i, sim_user_id in enumerate(similar_user_list):
            try:
                title_list = [
                    meta_name_dict["title"][str(item_id)]
                    for item_id in data[sim_user_id][-(self.max_items + 1) :]
                ]
            except:
                title_list = [
                    meta_name_dict["title"][str(item_id)]
                    for item_id in self.train[sim_user_id][-(self.max_items + 1) :]
                ]
            user_texts.append(f"{i}. <user {sim_user_id}>: {title_list}")

        query += "\n".join(user_texts)
        query += f"Please select the most related items based on the similar users of <user {user_id}>."

        # Cache the result
        self.cache[cache_key] = query
        return query

    def __len__(self) -> int:
        return len(self.data)

    def tokenize_batch(self, texts, max_length):
        """Batch tokenize multiple texts at once for efficiency"""
        return self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> Tuple:
        user_id = self.usermap[idx]

        # Generate or retrieve cached queries
        user_item_query = self._format_item_list_query(
            user_id, self.data, self.meta_name_dict
        )
        similar_users_query = self._format_similar_users_query(
            user_id, self.data, self.meta_name_dict, self.similar_users
        )

        target_item_id = self.data[user_id][-1]
        target_item_title = self.meta_name_dict["title"][str(target_item_id)]

        # Check if tokenized queries are already cached
        ui_cache_key = f"ui_tokenized_{user_id}"
        su_cache_key = f"su_tokenized_{user_id}"

        if ui_cache_key in self.cache:
            tokenized_ui_query = self.cache[ui_cache_key]
        else:
            tokenized_ui_query = self.tokenizer(
                user_item_query,
                return_tensors="pt",
                padding="max_length",
                max_length=self.config["max_length"],
                truncation=True,
            )
            self.cache[ui_cache_key] = tokenized_ui_query

        if su_cache_key in self.cache:
            tokenized_su_query = self.cache[su_cache_key]
        else:
            tokenized_su_query = self.tokenizer(
                similar_users_query,
                return_tensors="pt",
                padding="max_length",
                max_length=self.config["max_input_length"],
                truncation=True,
            )
            self.cache[su_cache_key] = tokenized_su_query

        return {
            "user_id": user_id,
            "user_item_ids": " ".join([str(i) for i in self.data[user_id][:-1]]),
            "ui_query_input_ids": tokenized_ui_query["input_ids"].squeeze(),
            "ui_query_attention_mask": tokenized_ui_query["attention_mask"].squeeze(),
            "su_query_input_ids": tokenized_su_query["input_ids"].squeeze(),
            "su_query_attention_mask": tokenized_su_query["attention_mask"].squeeze(),
            "target_item_id": target_item_id,
            "target_item_title": target_item_title,
        }
