import argparse
import os, sys
import json
import gc
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel


class TextDataset(Dataset):
    def __init__(self, text_list, idx_list, tokenizer, max_length=2048):
        print("Pre-tokenizing all texts...")
        self.encodings = tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.idxs = torch.tensor(idx_list, dtype=torch.long)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return {
            "input_ids": self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
            "idx": self.idxs[i],
        }


def process_large_dataset(
    text_list,
    iidx_list,
    model,
    tokenizer,
    device,
    dim=768,
    max_length=2048,
    batch_size=128,
    chunk_size=50000,
):

    total_size = len(text_list) + 1
    # Initialize as float16 to match autocast dtype
    all_embeddings = torch.zeros((total_size, dim), dtype=torch.float16)

    for chunk_start in range(0, total_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_size)

        print(
            f"Processing chunk {chunk_start//chunk_size + 1} of {(total_size-1)//chunk_size + 1}"
        )

        chunk_dataset = TextDataset(
            text_list[chunk_start:chunk_end],
            iidx_list[chunk_start:chunk_end],
            tokenizer,
            max_length=max_length,
        )

        dataloader = DataLoader(
            chunk_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model.eval()
        with torch.no_grad(), autocast():
            for batch in tqdm(dataloader):
                inputs = {
                    "input_ids": batch["input_ids"].to(device, non_blocking=True),
                    "attention_mask": batch["attention_mask"].to(
                        device, non_blocking=True
                    ),
                }

                outputs = model(**inputs).pooler_output
                # Store directly in half precision
                all_embeddings[batch["idx"]] = outputs.cpu()

        del chunk_dataset, dataloader
        gc.collect()
        torch.cuda.empty_cache()

    # Convert back to float32 at the end if needed
    all_embeddings = all_embeddings.float()
    return nn.Parameter(all_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/home/doyooni303/experiments/LLMRec/data/amazon/Books",
    )
    parser.add_argument("--fname", type=str, default="Books")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--model", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    torch.random.manual_seed(303)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    itemmap = json.load(
        open(os.path.join(args.path, f"{args.fname}_itemmap.json"), "r")
    )
    meta_name_dict = json.load(
        open(os.path.join(args.path, f"{args.fname}_meta_name_dict.json"), "r")
    )

    iidx_list, text_list = (
        [],
        [],
    )
    meta_keys = list(meta_name_dict.keys())
    for iid, idx in tqdm(itemmap.items()):
        item_text = ", ".join(
            [f"{key.upper()}: {meta_name_dict[key][str(idx)]}" for key in meta_keys]
        )
        iidx_list.append(idx)
        text_list.append(item_text)

    embeddings = process_large_dataset(
        text_list,
        iidx_list,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        chunk_size=50000,
    )

    torch.save(embeddings, os.path.join(args.path, f"{args.fname}_item_embeddings.pt"))
    print("Item Embeddings saved!")
