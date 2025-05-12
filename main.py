import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import yaml
import hashlib
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from src.models import CandiRec
from src.train import Trainer
from src.utils import set_seed

import wandb


def setup_distributed():
    """Initialize distributed training"""
    # Check if we're in distributed mode
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU mode
        world_size = 1
        rank = 0
        local_rank = 0
        
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    return local_rank, rank, world_size


def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main(config):
    # Setup distributed training
    local_rank, rank, world_size = setup_distributed()
    
    # Update config with distributed settings
    config.update({
        'local_rank': local_rank,
        'rank': rank,
        'world_size': world_size,
        'gpu': local_rank  # Use local_rank as GPU id
    })
    
    # Set seed for reproducibility
    set_seed(config["seed"] + rank)  # Different seed for each process
    
    # Initialize wandb only on main process
    number = config["max_epochs"] if config["train_mode"] == "epochs" else config["max_steps"]
    if rank == 0:
        wandb.init(
            project=f"CandiRec-{config['fname']}-{config['type']}",
            config=config,
            name=f"TopK-{config['top_k']}-GPU-{world_size}-{config['train_mode']}-{number}"
        )
    
    device = torch.device(f'cuda:{local_rank}')
    
    # Create model
    model = CandiRec(config).to(device=device, dtype=torch.bfloat16)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]))
    
    trainer = Trainer(model, optimizer, config)
    trainer.fit()
    
    if config["test"] and rank == 0:  # Run test only on main process
        trainer.test()
    
    # Cleanup
    if rank == 0:
        wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--type", type=str, default="Original")
    parser.add_argument("--fname", type=str, default="Books")
    parser.add_argument("--gpu", type=str, default="0,1,2,3", help="GPU ids to use (comma separated)")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--train_mode", type=str, default="epochs", help="[epochs, steps]")
    
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES based on gpu argument
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
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