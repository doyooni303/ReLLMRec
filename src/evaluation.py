# import numpy as np
# from typing import Union


# class Metric(object):
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.results = {}
#         self.batch_count = 0

#     def update(self, target_ids, max_k_ids, top_k_list):
#         scores = self.calculate_scores(target_ids, max_k_ids, top_k_list)
#         if self.results:
#             for K in top_k_list:
#                 self.results[K]["HR"] += scores[K]["HR"]
#                 self.results[K]["NDCG"] += scores[K]["NDCG"]
#         else:
#             self.results = scores
#         self.batch_count += 1

#     def fast_update(self, target_ids, max_k_ids, top_k):
#         scores = self.fast_calculate_scores(target_ids, max_k_ids, top_k)
#         if self.results:
#             self.results[f"HR@{top_k}"] += scores[f"HR@{top_k}"]
#         else:
#             self.results = scores
#         self.batch_count += 1

#     def calculate_scores(self, target_ids, max_k_ids, top_k_list: Union[list, int]):
#         measure = dict()
#         for K in top_k_list:
#             if K > max_k_ids.shape[1]:
#                 K = max_k_ids.shape[1]
#                 print(f"K is larger than the number of items. K is set to {K}")

#             top_k_ids = max_k_ids[:, :K]
#             hr = Metric.hit_ratio(target_ids, top_k_ids)
#             NDCG = Metric.NDCG(target_ids, top_k_ids, K)
#             measure.update({K: {"HR": hr, "NDCG": NDCG}})

#         return measure

#     def fast_calculate_scores(self, target_ids, max_k_ids, top_k: int):
#         measure = dict()
#         if top_k > max_k_ids.shape[1]:
#             top_k = max_k_ids.shape[1]
#             print(f"K is larger than the number of items. K is set to {top_k}")

#         top_k_ids = max_k_ids[:, :top_k]
#         hr = Metric.hit_ratio(target_ids, top_k_ids)
#         measure.update({f"HR@{top_k}": hr})
#         return measure

#     def get_results(self):
#         final_results = {}
#         for K in self.results.keys():
#             final_results[K] = {
#                 "HR": round(self.results[K]["HR"] / self.batch_count, 5),
#                 "NDCG": round(self.results[K]["NDCG"] / self.batch_count, 5),
#             }
#         return final_results

#     @staticmethod
#     def hits(target_ids, top_k_ids):
#         if len(top_k_ids.shape) == 1:
#             top_k_ids = top_k_ids.reshape(1, -1)
#         hit_count = 0
#         for target_id, top_k_id in zip(target_ids, top_k_ids):
#             if target_id in top_k_id:
#                 hit_count += 1
#         return hit_count

#     @staticmethod
#     def hit_ratio(target_ids, top_k_ids):
#         hit_count = Metric.hits(target_ids, top_k_ids)
#         return round(hit_count / len(target_ids), 5)

#     @staticmethod
#     def precision(taret_ids, top_k_ids, K):
#         hit_ratio = Metric.hit_ratio(taret_ids, top_k_ids)
#         return round(hit_ratio / K, 5)

#     @staticmethod
#     def recall(taret_ids, top_k_ids):
#         return Metric.hit_ratio(taret_ids, top_k_ids)

#     @staticmethod
#     def F1(taret_ids, top_k_ids, K):
#         prec = Metric.precision(taret_ids, top_k_ids, K)
#         recall = Metric.recall(taret_ids, top_k_ids)

#         if (prec + recall) != 0:
#             return round(2 * prec * recall / (prec + recall), 5)
#         else:
#             return 0

#     @staticmethod
#     def NDCG(target_ids, top_k_ids, K):
#         expanded_targets = np.expand_dims(target_ids, axis=1)
#         hit_matrix = (top_k_ids == expanded_targets).astype(np.float64)
#         position_indices = np.arange(1, K + 1)
#         discounts = 1 / np.log2(position_indices + 1)

#         dcg = np.sum(hit_matrix * discounts, axis=1)
#         idcg = 1 / np.log2(2)
#         ndcg = dcg / idcg

#         return ndcg.mean()


import torch
from typing import Union


class Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.results = {}
        self.batch_count = 0

    def update(self, target_ids, max_k_ids, top_k_list, dtype=None):
        # Convert to tensors if they aren't already
        if not isinstance(target_ids, torch.Tensor):
            target_ids = torch.tensor(target_ids, dtype=dtype)
        if not isinstance(max_k_ids, torch.Tensor):
            max_k_ids = torch.tensor(max_k_ids, dtype=dtype)

        # # Move to GPU if available
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # target_ids = target_ids.to(device)
        # max_k_ids = max_k_ids.to(device)

        scores = self.calculate_scores(target_ids, max_k_ids, top_k_list)
        if self.results:
            for K in top_k_list:
                self.results[K]["HR"] += scores[K]["HR"]
                self.results[K]["NDCG"] += scores[K]["NDCG"]
        else:
            self.results = scores
        self.batch_count += 1

    def fast_update(self, target_ids, max_k_ids, top_k, dtype=None):
        # Convert to tensors if they aren't already
        if not isinstance(target_ids, torch.Tensor):
            target_ids = torch.tensor(target_ids, dtype=dtype)
        if not isinstance(max_k_ids, torch.Tensor):
            max_k_ids = torch.tensor(max_k_ids, dtype=dtype)

        # # Move to GPU if available
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # target_ids = target_ids.to(device)
        # max_k_ids = max_k_ids.to(device)

        scores = self.fast_calculate_scores(target_ids, max_k_ids, top_k)
        if self.results:
            self.results[f"HR@{top_k}"] += scores[f"HR@{top_k}"]
        else:
            self.results = scores
        self.batch_count += 1

    def calculate_scores(self, target_ids, max_k_ids, top_k_list: Union[list, int]):
        measure = dict()
        for K in top_k_list:
            if K > max_k_ids.shape[1]:
                K = max_k_ids.shape[1]
                print(f"K is larger than the number of items. K is set to {K}")

            top_k_ids = max_k_ids[:, :K]
            hr = Metric.hit_ratio(target_ids, top_k_ids)
            NDCG = Metric.NDCG(target_ids, top_k_ids, K)
            measure.update({K: {"HR": hr, "NDCG": NDCG}})

        return measure

    def fast_calculate_scores(self, target_ids, max_k_ids, top_k: int):
        measure = dict()
        if top_k > max_k_ids.shape[1]:
            top_k = max_k_ids.shape[1]
            print(f"K is larger than the number of items. K is set to {top_k}")

        top_k_ids = max_k_ids[:, :top_k]
        hr = Metric.hit_ratio(target_ids, top_k_ids)
        measure.update({f"HR@{top_k}": hr})
        return measure

    def get_results(self):
        final_results = {}
        for K in self.results.keys():
            if isinstance(K, int):
                final_results[K] = {
                    "HR": round(self.results[K]["HR"] / self.batch_count, 5),
                    "NDCG": round(self.results[K]["NDCG"] / self.batch_count, 5),
                }
            else:  # For 'HR@K' keys
                final_results[K] = round(self.results[K] / self.batch_count, 5)
        return final_results

    @staticmethod
    def hit_ratio(target_ids, top_k_ids):
        # Expand targets to match shape for comparison [batch_size, 1]
        expanded_targets = target_ids.unsqueeze(1)

        # Count hits with vectorized operations
        hits = (top_k_ids == expanded_targets).any(dim=1).sum().item()
        return round(hits / len(target_ids), 5)

    @staticmethod
    def precision(target_ids, top_k_ids, K):
        hit_ratio = Metric.hit_ratio(target_ids, top_k_ids)
        return round(hit_ratio / K, 5)

    @staticmethod
    def recall(target_ids, top_k_ids):
        return Metric.hit_ratio(target_ids, top_k_ids)

    @staticmethod
    def F1(target_ids, top_k_ids, K):
        prec = Metric.precision(target_ids, top_k_ids, K)
        recall = Metric.recall(target_ids, top_k_ids)

        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall), 5)
        else:
            return 0

    @staticmethod
    def NDCG(target_ids, top_k_ids, K):
        # Expand targets for comparison [batch_size, 1]
        expanded_targets = target_ids.unsqueeze(1)

        # Create hit matrix [batch_size, K]
        hit_matrix = (top_k_ids == expanded_targets).float()

        # Create position discounts [1, K]
        position_indices = torch.arange(1, K + 1, device=top_k_ids.device)
        discounts = 1.0 / torch.log2(position_indices + 1)

        # Calculate DCG [batch_size]
        dcg = torch.sum(hit_matrix * discounts, dim=1)

        # IDCG is constant for single-item recommendation (position 1)
        idcg = 1.0 / torch.log2(torch.tensor(2.0, device=top_k_ids.device))

        # Calculate NDCG [batch_size]
        ndcg = dcg / idcg

        # Return mean NDCG
        return ndcg.mean().item()
