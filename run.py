import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import yaml
import hashlib
import torch
from torch.optim import AdamW

from src.models import CandiRec
from src.train import Trainer
from src.utils import set_seed




def main(config):
    set_seed(config["seed"])

    device = torch.device(
        f'cuda:{config["gpu"]}' if torch.cuda.is_available() else "cpu"
    )

    model = CandiRec(config).to(device=device, dtype=torch.bfloat16)
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]))

    trainer = Trainer(model, optimizer, config)
    trainer.fit()

    if config["test"]:
        trainer.test()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--fname", type=str, default="Books")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--train_mode", type=str, default="epochs", help="[epochs, steps]")

    args = parser.parse_args()
    yaml_path = os.path.join("./configs", f"{args.fname}.yaml")
    config = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
    for key, value in args.__dict__.items():
        config.update({key: value})

    if config["train_mode"] == "epochs":
        name = f"-".join([f"{key}_{config[key]}" for key in ["top_k", "max_epochs"]])
        config.update({"eval_interval": config["max_epochs"] // 10})
    elif config["train_mode"] == "steps":
        name = f"-".join([f"{key}_{config[key]}" for key in ["top_k", "max_steps"]])
        config.update({"eval_interval": config["max_steps"] // 10})

    save_path = os.path.join(config["output"], config["fname"], name)
    os.makedirs(save_path, exist_ok=True)
    config.update({"save_path": save_path})

    main(config)
