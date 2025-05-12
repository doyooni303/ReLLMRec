# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from collections import OrderedDict
# from peft import get_peft_model, LoraConfig, TaskType
# from transformers import AutoModelForCausalLM


# class CandiRec(nn.Module):
#     def __init__(
#         self,
#         config: dict,
#         dtype: torch.dtype = torch.bfloat16,
#     ):
#         super(CandiRec, self).__init__()
#         torch.random.manual_seed(config["seed"])

#         self.config = config
#         self.top_k = config["top_k"]
#         self.max_k = max(config["top_k_list"])
#         self.dtype = dtype
#         self.device = torch.device(
#             f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu"
#         )

#         self._initialize_model_components()
#         self.llm = self._load_llm()

#     def _initialize_model_components(self):
#         # Load item embeddings and register as buffer (non-trainable) or parameter (trainable)
#         item_embeddings = torch.load(
#             os.path.join(
#                 self.config["path"], f"{self.config['fname']}_item_embeddings.pt"
#             )
#         )
#         self.item_embeddings = nn.Parameter(item_embeddings)

#         # Create alignment layers as proper nn.Module subcomponents
#         self.alignment = nn.Sequential(
#             nn.Linear(self.config["llm_dim"], self.config["latent_dim"]),
#             nn.Linear(self.config["latent_dim"], self.config["item_dim"]),
#         )

#         # Register additional parameters
#         self.linear = nn.Parameter(torch.randn(self.config["item_dim"]))

#     def _load_llm(
#         self,
#     ):
#         # llm = AutoModelForCausalLM.from_pretrained(
#         #     self.config["model_name"], torch_dtype=self.dtype, device_map=self.device
#         # )
#         llm = AutoModelForCausalLM.from_pretrained(
#             self.config["model_name"], device_map=self.device, torch_dtype=self.dtype
#         )
#         lora_config = LoraConfig(
#             task_type=getattr(TaskType, self.config["LoRA TaskType"]),
#             **self.config["LoRA"],
#         )
#         llm = get_peft_model(llm, lora_config)
#         return llm

#     def find_top_k_item_embeddings(
#         self, query_vectors, item_embeddings, batch_size=10000, k=5
#     ):
#         # Ensure inputs are on the correct device
#         if query_vectors.device != item_embeddings.device:
#             query_vectors = query_vectors.to(self.device)
#             item_embeddings = item_embeddings.to(self.device)

#         # Normalize the vectors for cosine similarity
#         query_vectors_normalized = F.normalize(query_vectors, p=2, dim=1).to(
#             dtype=self.dtype
#         )
#         item_embeddings_normalized = F.normalize(item_embeddings, p=2, dim=1).to(
#             dtype=self.dtype
#         )

#         # For large embedding matrices, compute similarity in batches
#         num_batches = (item_embeddings.shape[0] + batch_size - 1) // batch_size

#         # Initialize variables to store results on the GPU
#         num_queries = query_vectors.shape[0]
#         top_k_similarities = torch.full(
#             (num_queries, k), -float("inf"), device=self.device, dtype=self.dtype
#         )
#         top_k_indices = torch.zeros(
#             (num_queries, k),
#             device=self.device,
#             dtype=torch.long,
#         )

#         # Compute similarities batch by batch
#         for i in range(num_batches):
#             start_idx = i * batch_size
#             end_idx = min((i + 1) * batch_size, item_embeddings.shape[0])

#             # Compute cosine similarity for this batch
#             batch_similarities = torch.matmul(
#                 query_vectors_normalized,
#                 item_embeddings_normalized[start_idx:end_idx].t(),
#             )

#             # If index 0 is in this batch, set its similarity to -inf to ignore it
#             if start_idx == 0 and end_idx > 0:
#                 batch_similarities[:, 0] = -float("inf")

#             # For each query, get top-k similarities from this batch
#             batch_top_values, batch_top_indices = torch.topk(
#                 batch_similarities, min(k, batch_similarities.shape[1]), dim=1
#             )

#             # Adjust indices to account for batching
#             batch_top_indices = batch_top_indices + start_idx

#             # Merge with existing top-k
#             if i == 0:
#                 # For the first batch, just store the values
#                 top_k_similarities[:, : batch_top_values.shape[1]] = batch_top_values
#                 top_k_indices[:, : batch_top_indices.shape[1]] = batch_top_indices
#             else:
#                 # For subsequent batches, merge and re-sort
#                 combined_similarities = torch.cat(
#                     [top_k_similarities, batch_top_values], dim=1
#                 )
#                 combined_indices = torch.cat([top_k_indices, batch_top_indices], dim=1)

#                 # Re-sort to get the overall top-k
#                 for q in range(num_queries):
#                     q_top_values, q_top_indices = torch.topk(
#                         combined_similarities[q], k, dim=0
#                     )
#                     top_k_similarities[q] = q_top_values
#                     top_k_indices[q] = combined_indices[q][q_top_indices]

#             # Optional: Clear GPU cache periodically
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         # Final check to ensure index 0 isn't in the results
#         for q in range(num_queries):
#             # Check if index 0 is in the results
#             zero_mask = top_k_indices[q] == 0
#             if torch.any(zero_mask):
#                 # If found, replace with the next best item
#                 remaining_indices = torch.ones(
#                     item_embeddings.shape[0], dtype=torch.bool, device=self.device
#                 )
#                 # Mark indices already in top-k as used
#                 remaining_indices[top_k_indices[q]] = False
#                 # Also mark index 0 as used
#                 remaining_indices[0] = False

#                 # Find the count of indices to replace
#                 num_to_replace = torch.sum(zero_mask).item()

#                 if num_to_replace > 0 and torch.any(remaining_indices):
#                     # Find remaining embeddings
#                     remaining_embeddings = item_embeddings_normalized[remaining_indices]

#                     # Compute similarities for unused indices
#                     remaining_similarities = torch.matmul(
#                         query_vectors_normalized[q : q + 1], remaining_embeddings.t()
#                     )[0]

#                     # Get top replacements
#                     replacement_values, replacement_relative_indices = torch.topk(
#                         remaining_similarities,
#                         min(num_to_replace, torch.sum(remaining_indices).item()),
#                     )

#                     # Convert relative indices to absolute
#                     replacement_indices = torch.nonzero(
#                         remaining_indices, as_tuple=True
#                     )[0][replacement_relative_indices]

#                     # Replace zeros in the results
#                     zero_positions = torch.nonzero(zero_mask, as_tuple=True)[0]
#                     for j in range(min(len(zero_positions), len(replacement_indices))):
#                         top_k_indices[q, zero_positions[j]] = replacement_indices[j]

#         # Retrieve the actual embeddings using the indices
#         # Shape: [batch_size, k, embedding_dim]
#         top_k_item_embeddings = torch.zeros(
#             (num_queries, k, item_embeddings.shape[1]),
#             device=self.device,
#             dtype=self.dtype,
#         )
#         for q in range(num_queries):
#             top_k_item_embeddings[q] = item_embeddings[top_k_indices[q]]

#         return top_k_indices, top_k_item_embeddings

#     def forward(self, batch: dict, flag: str = "train", target_include: bool = True):
#         torch.random.manual_seed(self.config["seed"])
#         batch_size = batch["target_item_id"].shape[0]
#         # Target Item embeddings: [batch_size, item_dim]
#         target_item_embeddings = self.item_embeddings[batch["target_item_id"]]

#         # User embeddings: [batch_size, item_dim]
#         user_embeddings = torch.zeros(
#             batch_size,
#             # self.config["batch_size"],
#             self.config["item_dim"],
#         ).to(self.device)
#         for i, ids in enumerate(batch["user_item_ids"]):
#             item_ids = [int(iid) for iid in ids.split(" ")]
#             user_embeddings[i] += self.item_embeddings[item_ids].mean(dim=0)

#         # Get Query Vectors
#         item_history_query = self.llm(
#             input_ids=batch["ui_query_input_ids"],
#             attention_mask=batch["ui_query_attention_mask"],
#             output_hidden_states=True,
#         ).hidden_states[-1][:, -1, :]
#         similar_user_query = self.llm(
#             input_ids=batch["su_query_attention_mask"],
#             attention_mask=batch["su_query_attention_mask"],
#             output_hidden_states=True,
#         ).hidden_states[-1][:, -1, :]

#         query_embedding = self.alignment(item_history_query + similar_user_query)

#         # Get Item Embeddings
#         self.max_k_indices, top_k_item_embeddings = self.find_top_k_item_embeddings(
#             query_embedding, self.item_embeddings, k=self.max_k
#         )

#         max_k_indices = self.max_k_indices.clone()
#         self.top_k_indices = max_k_indices[:, : self.top_k]
#         # self.max_k_indices = self.max_k_indices.detach().cpu().numpy()

#         # Ensure target item is in the top-k
#         if (flag == "train") or target_include:
#             for i, target in enumerate(batch["target_item_id"]):
#                 if target in self.top_k_indices[i]:
#                     pass
#                 else:
#                     self.top_k_indices[i][-1] = target

#         perm_cols = torch.randperm(self.config["top_k"])
#         perm_k_indices = self.top_k_indices[:, perm_cols]
#         perm_k_item_embeddings = top_k_item_embeddings[:, perm_cols, :]

#         # Get Target Item positions

#         # target_positions = torch.empty(
#         #     self.config["batch_size"], dtype=torch.long, device=self.device
#         # )
#         # non_target_positions = torch.ones(
#         #     (self.config["batch_size"], self.config["top_k"]),
#         #     dtype=torch.bool,
#         #     device=self.device,
#         # )

#         target_positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
#         non_target_positions = torch.ones(
#             (batch_size, self.config["top_k"]),
#             dtype=torch.bool,
#             device=self.device,
#         )

#         for i, (target_id, indices) in enumerate(
#             zip(batch["target_item_id"], perm_k_indices)
#         ):
#             position = torch.where(target_id == indices)[0].item()
#             target_positions[i] = position
#             non_target_positions[i][position] = False

#         self.logits = torch.matmul(perm_k_item_embeddings, self.linear)
#         self.top_k_indices = self.top_k_indices.detach().cpu().numpy()

#         if flag == "train":
#             ce_loss = F.cross_entropy(self.logits, target_positions)

#             # Get Contrastive loss
#             negative_item_embeddings = (
#                 perm_k_item_embeddings[non_target_positions]
#                 .reshape(batch_size, -1, self.config["item_dim"])
#                 .transpose(1, 2)
#             )

#             # Get Cross Entropy Loss
#             cl_loss = (
#                 1
#                 - F.sigmoid(
#                     torch.bmm(
#                         target_item_embeddings.unsqueeze(1), negative_item_embeddings
#                     ).squeeze(1)
#                 )
#             ).mean(1)

#             self.loss = ce_loss + cl_loss

#             return self.top_k_indices, self.loss.mean()

#         else:
#             idxs = self.logits.max(dim=1)[1].detach().cpu().numpy()
#             rows = np.arange(self.top_k_indices.shape[0])
#             self.preds = self.top_k_indices[rows, idxs]

#             return self.top_k_indices, self.preds

#     def save(self, path: str):
#         torch.save(self.state_dict(), path)

#     def load(self, path: str):
#         self.load_state_dict(torch.load(path))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM


class CandiRec(nn.Module):
    def __init__(
        self,
        config: dict,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super(CandiRec, self).__init__()
        torch.random.manual_seed(config["seed"])

        self.config = config
        self.top_k = config["top_k"]
        self.max_k = max(config["top_k_list"])
        self.dtype = dtype
        
        # Use gpu from config (which will be local_rank in DDP)
        self.device = torch.device(
            f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu"
        )

        self._initialize_model_components()
        self.llm = self._load_llm()

    def _initialize_model_components(self):
        # Load item embeddings and register as buffer (non-trainable) or parameter (trainable)
        item_embeddings = torch.load(
            os.path.join(
                self.config["path"], f"{self.config['fname']}_item_embeddings.pt"
            ),
            map_location=self.device  # Ensure embeddings are loaded to correct device
        )
        self.item_embeddings = nn.Parameter(item_embeddings)

        # Create alignment layers as proper nn.Module subcomponents
        self.alignment = nn.Sequential(
            nn.Linear(self.config["llm_dim"], self.config["latent_dim"]),
            nn.Linear(self.config["latent_dim"], self.config["item_dim"]),
        )

        # Register additional parameters
        self.linear = nn.Parameter(torch.randn(self.config["item_dim"]))

    def _load_llm(self):
        # Load LLM with device map for the specific GPU
        llm = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"], 
            device_map={"": self.device},  # Map to specific device
            torch_dtype=self.dtype
        )
        lora_config = LoraConfig(
            task_type=getattr(TaskType, self.config["LoRA TaskType"]),
            **self.config["LoRA"],
        )
        llm = get_peft_model(llm, lora_config)
        return llm

    def find_top_k_item_embeddings(
        self, query_vectors, item_embeddings, batch_size=10000, k=5
    ):
        # Ensure inputs are on the correct device
        if query_vectors.device != item_embeddings.device:
            query_vectors = query_vectors.to(self.device)
            item_embeddings = item_embeddings.to(self.device)

        # Normalize the vectors for cosine similarity
        query_vectors_normalized = F.normalize(query_vectors, p=2, dim=1).to(
            dtype=self.dtype
        )
        item_embeddings_normalized = F.normalize(item_embeddings, p=2, dim=1).to(
            dtype=self.dtype
        )

        # For large embedding matrices, compute similarity in batches
        num_batches = (item_embeddings.shape[0] + batch_size - 1) // batch_size

        # Initialize variables to store results on the GPU
        num_queries = query_vectors.shape[0]
        top_k_similarities = torch.full(
            (num_queries, k), -float("inf"), device=self.device, dtype=self.dtype
        )
        top_k_indices = torch.zeros(
            (num_queries, k),
            device=self.device,
            dtype=torch.long,
        )

        # Compute similarities batch by batch
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, item_embeddings.shape[0])

            # Compute cosine similarity for this batch
            batch_similarities = torch.matmul(
                query_vectors_normalized,
                item_embeddings_normalized[start_idx:end_idx].t(),
            )

            # If index 0 is in this batch, set its similarity to -inf to ignore it
            if start_idx == 0 and end_idx > 0:
                batch_similarities[:, 0] = -float("inf")

            # For each query, get top-k similarities from this batch
            batch_top_values, batch_top_indices = torch.topk(
                batch_similarities, min(k, batch_similarities.shape[1]), dim=1
            )

            # Adjust indices to account for batching
            batch_top_indices = batch_top_indices + start_idx

            # Merge with existing top-k
            if i == 0:
                # For the first batch, just store the values
                top_k_similarities[:, : batch_top_values.shape[1]] = batch_top_values
                top_k_indices[:, : batch_top_indices.shape[1]] = batch_top_indices
            else:
                # For subsequent batches, merge and re-sort
                combined_similarities = torch.cat(
                    [top_k_similarities, batch_top_values], dim=1
                )
                combined_indices = torch.cat([top_k_indices, batch_top_indices], dim=1)

                # Re-sort to get the overall top-k
                for q in range(num_queries):
                    q_top_values, q_top_indices = torch.topk(
                        combined_similarities[q], k, dim=0
                    )
                    top_k_similarities[q] = q_top_values
                    top_k_indices[q] = combined_indices[q][q_top_indices]

            # Optional: Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final check to ensure index 0 isn't in the results
        for q in range(num_queries):
            # Check if index 0 is in the results
            zero_mask = top_k_indices[q] == 0
            if torch.any(zero_mask):
                # If found, replace with the next best item
                remaining_indices = torch.ones(
                    item_embeddings.shape[0], dtype=torch.bool, device=self.device
                )
                # Mark indices already in top-k as used
                remaining_indices[top_k_indices[q]] = False
                # Also mark index 0 as used
                remaining_indices[0] = False

                # Find the count of indices to replace
                num_to_replace = torch.sum(zero_mask).item()

                if num_to_replace > 0 and torch.any(remaining_indices):
                    # Find remaining embeddings
                    remaining_embeddings = item_embeddings_normalized[remaining_indices]

                    # Compute similarities for unused indices
                    remaining_similarities = torch.matmul(
                        query_vectors_normalized[q : q + 1], remaining_embeddings.t()
                    )[0]

                    # Get top replacements
                    replacement_values, replacement_relative_indices = torch.topk(
                        remaining_similarities,
                        min(num_to_replace, torch.sum(remaining_indices).item()),
                    )

                    # Convert relative indices to absolute
                    replacement_indices = torch.nonzero(
                        remaining_indices, as_tuple=True
                    )[0][replacement_relative_indices]

                    # Replace zeros in the results
                    zero_positions = torch.nonzero(zero_mask, as_tuple=True)[0]
                    for j in range(min(len(zero_positions), len(replacement_indices))):
                        top_k_indices[q, zero_positions[j]] = replacement_indices[j]

        # Retrieve the actual embeddings using the indices
        # Shape: [batch_size, k, embedding_dim]
        top_k_item_embeddings = torch.zeros(
            (num_queries, k, item_embeddings.shape[1]),
            device=self.device,
            dtype=self.dtype,
        )
        for q in range(num_queries):
            top_k_item_embeddings[q] = item_embeddings[top_k_indices[q]]

        return top_k_indices, top_k_item_embeddings

    def forward(self, batch: dict, flag: str = "train", target_include: bool = True):
        torch.random.manual_seed(self.config["seed"])
        batch_size = batch["target_item_id"].shape[0]
        # Target Item embeddings: [batch_size, item_dim]
        target_item_embeddings = self.item_embeddings[batch["target_item_id"]]

        # User embeddings: [batch_size, item_dim]
        user_embeddings = torch.zeros(
            batch_size,
            self.config["item_dim"],
        ).to(self.device)
        for i, ids in enumerate(batch["user_item_ids"]):
            item_ids = [int(iid) for iid in ids.split(" ")]
            user_embeddings[i] += self.item_embeddings[item_ids].mean(dim=0)

        # Get Query Vectors
        item_history_query = self.llm(
            input_ids=batch["ui_query_input_ids"],
            attention_mask=batch["ui_query_attention_mask"],
            output_hidden_states=True,
        ).hidden_states[-1][:, -1, :]
        similar_user_query = self.llm(
            input_ids=batch["su_query_attention_mask"],
            attention_mask=batch["su_query_attention_mask"],
            output_hidden_states=True,
        ).hidden_states[-1][:, -1, :]

        query_embedding = self.alignment(item_history_query + similar_user_query)

        # Get Item Embeddings
        self.max_k_indices, top_k_item_embeddings = self.find_top_k_item_embeddings(
            query_embedding, self.item_embeddings, k=self.max_k
        )

        max_k_indices = self.max_k_indices.clone()
        self.top_k_indices = max_k_indices[:, : self.top_k]

        # Ensure target item is in the top-k
        if (flag == "train") or target_include:
            for i, target in enumerate(batch["target_item_id"]):
                if target in self.top_k_indices[i]:
                    pass
                else:
                    self.top_k_indices[i][-1] = target

        perm_cols = torch.randperm(self.config["top_k"], device=self.device)  # Ensure same device
        perm_k_indices = self.top_k_indices[:, perm_cols]
        perm_k_item_embeddings = top_k_item_embeddings[:, perm_cols, :]

        # Get Target Item positions

        target_positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        non_target_positions = torch.ones(
            (batch_size, self.config["top_k"]),
            dtype=torch.bool,
            device=self.device,
        )

        for i, (target_id, indices) in enumerate(
            zip(batch["target_item_id"], perm_k_indices)
        ):
            position = torch.where(target_id == indices)[0].item()
            target_positions[i] = position
            non_target_positions[i][position] = False

        self.logits = torch.matmul(perm_k_item_embeddings, self.linear)
        self.top_k_indices = self.top_k_indices.detach().cpu().numpy()

        if flag == "train":
            ce_loss = F.cross_entropy(self.logits, target_positions)

            # Get Contrastive loss
            negative_item_embeddings = (
                perm_k_item_embeddings[non_target_positions]
                .reshape(batch_size, -1, self.config["item_dim"])
                .transpose(1, 2)
            )

            # Get Cross Entropy Loss
            cl_loss = (
                1
                - F.sigmoid(
                    torch.bmm(
                        target_item_embeddings.unsqueeze(1), negative_item_embeddings
                    ).squeeze(1)
                )
            ).mean(1)

            self.loss = ce_loss + cl_loss

            return self.top_k_indices, self.loss.mean()

        else:
            idxs = self.logits.max(dim=1)[1].detach().cpu().numpy()
            rows = np.arange(self.top_k_indices.shape[0])
            self.preds = self.top_k_indices[rows, idxs]

            return self.top_k_indices, self.preds

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))