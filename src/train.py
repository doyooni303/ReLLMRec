# import json
# from tqdm import tqdm

# import torch
# from torch.utils.data import DataLoader

# from src.dataset import AmazonDataset
# from src.utils import Log
# from src.evaluation import Metric


# class Trainer(object):
#     def __init__(self, model, optimizer, config):
#         self.config = config
#         self.model = model
#         self.optimizer = optimizer
#         self.trainloader = DataLoader(
#             AmazonDataset(config, "train"),
#             batch_size=config["batch_size"],
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True,
#         )
#         self.validloader = DataLoader(
#             AmazonDataset(config, "valid"),
#             batch_size=config["batch_size"],
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True,
#         )

#         # Initiating best model and best score
#         self.min_k = min(self.config["top_k_list"])
#         self.best_model = None
#         self.best_results = None
#         self.best_score = 0

#         # Use f-string for better readability
#         self.log = Log(self.config)
#         self.device = torch.device(
#             f'cuda:{config["gpu"]}' if torch.cuda.is_available() else "cpu"
#         )
#         self.metric = Metric()

#     def to_device(self, batch, device):
#         for k, v in batch.items():
#             if ("input_ids" in k) or ("attention_mask" in k) or ("target_item_id" in k):
#                 batch[k] = v.to(device)
#             else:
#                 batch[k] = v
#         return batch

#     def train_eval_steps(self, model, optimizer, train_loader, valid_loader, steps):
#         model.train()
#         current_step = 0
#         while current_step < steps:
#             for i, batch in enumerate(train_loader):
#                 batch = self.to_device(batch, self.device)
#                 optimizer.zero_grad()
#                 _, loss = model(batch, flag="train")

#                 loss.backward()
#                 optimizer.step()
#                 current_step += 1
#                 self.log.add(
#                     f"Steps: {current_step}/{steps} || Iter: {i}/{len(train_loader)} || Loss: {loss.item()}"
#                 )

#                 if current_step % self.config["eval_interval"] == 0:
#                     # results = self.evaluate(model, valid_loader, self.metric)
#                     results = self.fast_evaluate(model, valid_loader, self.metric)
#                     self.log.add(
#                         f"Validation|| Steps: {current_step}/{steps} || Results: {results}"
#                     )

#                     # Save the best model based on validation results
#                     # if results[self.min_k]["NDCG"] > self.best_score:
#                     #     self.best_score = results[self.min_k]["NDCG"]
#                     #     self.best_model = model
#                     #     self.best_results = results
#                     #     self.save()

#                     if results.get(f"HR@{self.min_k}") > self.best_score:
#                         self.best_score = results.get(f"HR@{self.min_k}")
#                         self.best_model = model
#                         self.best_results = results
#                         self.save()

#                 if current_step >= steps:
#                     break

#             # Check if steps completed after the current epoch
#             if current_step >= steps:
#                 break

#         return model, optimizer

#     def train_epoch(self, model, optimizer, loader, epoch):
#         # losses = []
#         for i, batch in tqdm(enumerate(loader), desc="Training", total=len(loader)):
#             batch = self.to_device(batch, self.device)
#             optimizer.zero_grad()
#             _, loss = model(batch, flag="train")

#             loss.backward()
#             optimizer.step()
#             # losses.append(loss.item())
#             self.log.add(
#                 f"Epoch: {epoch}/{self.config['max_epochs']} || Iter: {i}/{len(loader)} || Loss: {loss.item()}"
#             )

#         return model, optimizer

#     def evaluate(self, model, loader, metric):
#         model.eval()
#         metric.reset()

#         for batch in tqdm(loader, desc="Evaluating"):
#             batch = self.to_device(batch, self.device)

#             top_k_indices, preds = model(batch, flag="valid")
#             max_k_indices = model.max_k_indices
#             # target_ids = batch["target_item_id"].numpy()

#             metric.update(
#                 batch["target_item_id"],
#                 max_k_indices,
#                 self.config["top_k_list"],
#                 dtype=max_k_indices.dtype,
#             )

#         return metric.get_results()

#     def fast_evaluate(self, model, loader, metric):
#         model.eval()
#         metric.reset()

#         for batch in tqdm(loader, desc="Evaluating"):
#             batch = self.to_device(batch, self.device)

#             top_k_indices, preds = model(batch, flag="valid")
#             max_k_indices = model.max_k_indices
#             # target_ids = batch["target_item_id"].numpy()
#             # metric.update(target_ids, max_k_indices, self.config["top_k_list"])
#             # metric.fast_update(target_ids, max_k_indices, self.min_k)
#             metric.fast_update(
#                 batch["target_item_id"],
#                 max_k_indices,
#                 self.min_k,
#                 dtype=max_k_indices.dtype,
#             )

#         return metric.get_results()

#     def save(
#         self,
#     ):
#         torch.save(
#             self.best_model.state_dict(), f"{self.config['save_path']}/best_model.pth"
#         )
#         self.log.add("Model is saved")

#     def fit(
#         self,
#     ):
#         self.log.add("Start training")
#         self.model.train()
#         if self.config["train_mode"] == "epochs":
#             for epoch in range(self.config["max_epochs"]):
#                 self.model, self.optimizer = self.train_epoch(
#                     self.model, self.optimizer, self.trainloader, epoch
#                 )

#                 if (epoch + 1) % self.config["eval_interval"] == 0:
#                     results = self.fast_evaluate(
#                         self.model, self.validloader, self.metric
#                     )
#                     self.log.add(
#                         f"Validation|| Epoch: {epoch}/{self.config['max_epochs']} || Results: {results}"
#                     )

#                     # if results[self.min_k]["NDCG"] > self.best_score:
#                     #     self.best_score = results[self.min_k]["NDCG"]
#                     #     self.best_model = self.model
#                     #     self.best_results = results
#                     #     self.save()

#                     if results.get(f"HR@{self.min_k}") > self.best_score:
#                         self.best_score = results.get(f"HR@{self.min_k}")
#                         self.best_model = self.model
#                         self.best_results = results
#                         self.save()

#         elif self.config["train_mode"] == "steps":
#             # Implement the step-based training using the train_eval_steps method
#             self.model, self.optimizer = self.train_eval_steps(
#                 self.model,
#                 self.optimizer,
#                 self.trainloader,
#                 self.validloader,
#                 self.config["max_steps"],
#             )

#         self.log.add(f"Best results: {self.best_results}")

#     def test(
#         self,
#     ):
#         self.model.load_state_dict(
#             torch.load(f"{self.config['save_path']}/best_model.pth")
#         )
#         self.model.eval()

#         testloader = DataLoader(
#             AmazonDataset(self.config, "test"),
#             batch_size=self.config["batch_size"],
#             shuffle=False,
#             num_workers=4,
#         )

#         results = self.evaluate(self.model, testloader, self.metric)
#         self.log.add(f"Test results: {results}")

#         json.dumps(
#             results,
#             open(f"{self.config['save_path']}/test_results.json", "w"),
#             indent=4,
#         )
#         return results

import json
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.dataset import AmazonDataset
from src.utils import Log
from src.evaluation import Metric

import wandb


class Trainer(object):
    def __init__(self, model, optimizer, config):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        
        # Check if we're using DDP
        self.use_ddp = self.config.get('world_size', 1) > 1
        self.rank = self.config.get('rank', 0)
        self.world_size = self.config.get('world_size', 1)
        
        # Create dataset
        train_dataset = AmazonDataset(config, "train")
        valid_dataset = AmazonDataset(config, "valid")
        
        # Create samplers for DDP
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        ) if self.use_ddp else None
        
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        ) if self.use_ddp else None
        
        # Create dataloaders
        self.trainloader = DataLoader(
            train_dataset,
            batch_size=config["train_batch_size"],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True  # For DDP consistency
        )
        
        self.validloader = DataLoader(
            valid_dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            sampler=valid_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Store samplers for epoch updates
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        
        # Initiating best model and best score
        self.min_k = min(self.config["top_k_list"])
        self.best_model = None
        self.best_results = None
        self.best_score = 0
        
        # Use f-string for better readability
        self.log = Log(self.config)
        self.device = torch.device(
            f'cuda:{config["gpu"]}' if torch.cuda.is_available() else "cpu"
        )
        self.metric = Metric()
    
    def reduce_tensor(self, tensor):
        """Reduce tensor across all GPUs"""
        if not self.use_ddp:
            return tensor
        
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt
    
    def gather_results(self, results):
        """Gather results from all processes"""
        if not self.use_ddp:
            return results
        
        # Gather results from all processes
        gathered_results = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_results, results)
        
        # Average results on main process
        if self.rank == 0:
            final_results = {}
            for key in results.keys():
                if isinstance(results[key], dict):
                    final_results[key] = {}
                    for subkey in results[key].keys():
                        values = [r[key][subkey] for r in gathered_results]
                        final_results[key][subkey] = sum(values) / len(values)
                else:
                    values = [r[key] for r in gathered_results]
                    final_results[key] = sum(values) / len(values)
            return final_results
        return results
    
    def to_device(self, batch, device):
        for k, v in batch.items():
            if ("input_ids" in k) or ("attention_mask" in k) or ("target_item_id" in k):
                batch[k] = v.to(device)
            else:
                batch[k] = v
        return batch
    
    def train_eval_steps(self, model, optimizer, train_loader, valid_loader, steps):
        model.train()
        current_step = 0
        total_loss = 0
        step_count = 0
        # Initialize tqdm
        pbar = tqdm(total=steps, desc="Training", position=0, leave=True)

        while current_step < steps:
            for i, batch in enumerate(train_loader):
                batch = self.to_device(batch, self.device)
                optimizer.zero_grad()
                
                # Handle DDP model
                if self.use_ddp:
                    _, loss = self.model.module(batch, flag="train")
                else:
                    _, loss = model(batch, flag="train")
                
                loss.backward()
                optimizer.step()
                current_step += 1
                
                # Reduce loss across GPUs
                reduced_loss = self.reduce_tensor(loss).item()
                total_loss += reduced_loss
                step_count += 1

                if self.rank == 0:
                    self.log.add(
                        f"Steps: {current_step}/{steps} || Iter: {i}/{len(train_loader)} || Loss: {reduced_loss}"
                    )
                    pbar.update(1)

                    # Log to wandb
                    wandb.log({
                        "train/loss": reduced_loss,
                        "train/step": current_step
                    })
                
                if current_step % self.config["eval_interval"] == 0:
                    results = self.fast_evaluate(model, valid_loader, self.metric)
                    results = self.gather_results(results)  # Gather results from all processes
                    
                    if self.rank == 0:
                        self.log.add(
                            f"Validation|| Steps: {current_step}/{steps} || Results: {results}"
                        )
                        
                        # Log validation metrics to wandb
                        wandb.log({
                            f"val/HR@{self.min_k}": results.get(f"HR@{self.min_k}"),
                            "val/step": current_step
                        })
                        
                        # Save the best model based on validation results
                        if results.get(f"HR@{self.min_k}") > self.best_score:
                            self.best_score = results.get(f"HR@{self.min_k}")
                            self.best_model = model
                            self.best_results = results
                            self.save()
                
                if current_step >= steps:
                    break
            
            # Check if steps completed after the current epoch
            if current_step >= steps:
                break
        
        # Log average training loss for the session
        if self.rank == 0 and step_count > 0:
            avg_loss = total_loss / step_count
            wandb.log({"train/avg_loss": avg_loss})
        
        return model, optimizer
    
    def train_epoch(self, model, optimizer, loader, epoch):
        # Set epoch for distributed sampler
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        total_loss = 0
        step_count = 0
        
        for i, batch in tqdm(enumerate(loader), desc="Training", total=len(loader)):
            batch = self.to_device(batch, self.device)
            optimizer.zero_grad()
            
            # Handle DDP model
            if self.use_ddp:
                _, loss = self.model.module(batch, flag="train")
            else:
                _, loss = model(batch, flag="train")
            
            loss.backward()
            optimizer.step()
            
            # Reduce loss across GPUs
            reduced_loss = self.reduce_tensor(loss).item()
            total_loss += reduced_loss
            step_count += 1
            
            if self.rank == 0:
                self.log.add(
                    f"Epoch: {epoch}/{self.config['max_epochs']} || Iter: {i}/{len(loader)} || Loss: {reduced_loss}"
                )
                # Log to wandb
                wandb.log({
                    "train/loss": reduced_loss,
                    "train/step": step_count
                })
        
        # Log average loss for the epoch
        if self.rank == 0 and step_count > 0:
            avg_loss = total_loss / step_count
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch": epoch
            })
        
        return model, optimizer
    
    def evaluate(self, model, loader, metric):
        model.eval()
        metric.reset()
        
        # Set epoch for distributed sampler
        if self.valid_sampler is not None:
            self.valid_sampler.set_epoch(0)
        
        for batch in tqdm(loader, desc="Evaluating"):
            batch = self.to_device(batch, self.device)
            
            # Handle DDP model
            if self.use_ddp:
                top_k_indices, preds = self.model.module(batch, flag="valid")
                max_k_indices = self.model.module.max_k_indices
            else:
                top_k_indices, preds = model(batch, flag="valid")
                max_k_indices = model.max_k_indices
            
            metric.update(
                batch["target_item_id"],
                max_k_indices,
                self.config["top_k_list"],
                dtype=max_k_indices.dtype,
            )
        
        results = metric.get_results()
        return self.gather_results(results)
    
    def fast_evaluate(self, model, loader, metric):
        model.eval()
        metric.reset()
        
        for batch in tqdm(loader, desc="Evaluating"):
            batch = self.to_device(batch, self.device)
            
            # Handle DDP model
            if self.use_ddp:
                top_k_indices, preds = self.model.module(batch, flag="valid")
                max_k_indices = self.model.module.max_k_indices
            else:
                top_k_indices, preds = model(batch, flag="valid")
                max_k_indices = model.max_k_indices
            
            metric.fast_update(
                batch["target_item_id"],
                max_k_indices,
                self.min_k,
                dtype=max_k_indices.dtype,
            )
        
        results = metric.get_results()
        return results  # Don't gather results here since we'll do it in the caller
    
    def save(self):
        # Only save on the main process
        if self.rank == 0:
            # Handle DDP model state dict
            if self.use_ddp:
                state_dict = self.best_model.module.state_dict()
            else:
                state_dict = self.best_model.state_dict()
            
            torch.save(
                state_dict, f"{self.config['save_path']}/best_model.pth"
            )
            self.log.add("Model is saved")
    
    def fit(self):
        if self.rank == 0:
            self.log.add("Start training")
            wandb.log({"train/start": 1})
        
        self.model.train()
        
        if self.config["train_mode"] == "epochs":
            for epoch in range(self.config["max_epochs"]):
                self.model, self.optimizer = self.train_epoch(
                    self.model, self.optimizer, self.trainloader, epoch
                )
                
                if (epoch + 1) % self.config["eval_interval"] == 0:
                    results = self.fast_evaluate(
                        self.model, self.validloader, self.metric
                    )
                    results = self.gather_results(results)  # Gather results from all processes
                    
                    if self.rank == 0:
                        self.log.add(
                            f"Validation|| Epoch: {epoch}/{self.config['max_epochs']} || Results: {results}"
                        )
                        
                        # Log validation metrics to wandb
                        wandb.log({
                            f"val/HR@{self.min_k}": results.get(f"HR@{self.min_k}"),
                            "val/epoch": epoch
                        })
                        
                        if results.get(f"HR@{self.min_k}") > self.best_score:
                            self.best_score = results.get(f"HR@{self.min_k}")
                            self.best_model = self.model
                            self.best_results = results
                            self.save()
        
        elif self.config["train_mode"] == "steps":
            # Implement the step-based training using the train_eval_steps method
            self.model, self.optimizer = self.train_eval_steps(
                self.model,
                self.optimizer,
                self.trainloader,
                self.validloader,
                self.config["max_steps"],
            )
        
        if self.rank == 0:
            self.log.add(f"Best results: {self.best_results}")
            wandb.log({"best_results": self.best_results})
    
    def test(self):
        # Only test on main process
        if self.rank != 0:
            return
        
        # Load best model
        state_dict = torch.load(f"{self.config['save_path']}/best_model.pth")
        
        # Handle DDP model
        if self.use_ddp:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        
        testloader = DataLoader(
            AmazonDataset(self.config, "test"),
            batch_size=self.config["eval_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        
        results = self.evaluate(self.model, testloader, self.metric)
        self.log.add(f"Test results: {results}")
        
        # Log test results to wandb
        test_results_wandb = {}
        for k, v in results.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    test_results_wandb[f"test/{k}_{sub_k}"] = sub_v
            else:
                test_results_wandb[f"test/{k}"] = v
        wandb.log(test_results_wandb)
        
        # Save results to JSON
        with open(f"{self.config['save_path']}/test_results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        return results