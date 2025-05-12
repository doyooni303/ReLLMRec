import os, sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import json
import heapq
import argparse
from multiprocessing import cpu_count
from typing import Dict, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import defaultdict

from data_preprocess import preprocess
from utils import data_partition
from src.utils import open_jsonl


def pruning(
    history: Dict[str, List[int]], min_length: int, max_length: int
) -> Dict[str, List[int]]:
    history_preprocessed = {}
    for key, value in tqdm(history.items(), desc="Preprocessing history"):
        if len(value) >= min_length:
            v = value[-max_length:]
            history_preprocessed.update({key: v})
    return history_preprocessed


class UserSimilarityFinder:
    def __init__(self, user_items: Dict):
        self.user_items = self.seq2set(user_items)
        self.inverted_index = self.create_inverted_index()

    @staticmethod
    def seq2set(user_items: Dict) -> Dict[str, Set[int]]:
        return {user_id: set(items) for user_id, items in user_items.items()}

    def create_inverted_index(self) -> Dict[int, Set[str]]:
        """Create an inverted index mapping items to users who have that item."""
        inverted_index = defaultdict(set)
        for user_id, items in self.user_items.items():
            for item in items:
                inverted_index[item].add(user_id)
        return inverted_index

    def find_similar_users(
        self, query_user_id: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find top-k similar users for a single query user."""
        query_items = self.user_items[query_user_id]

        # Get candidate users who share at least one item with query user
        candidates = set()
        for item in query_items:
            candidates.update(self.inverted_index[item])
        candidates.discard(query_user_id)

        # Calculate scores for candidates
        scores = []
        for candidate_id in candidates:
            candidate_items = self.user_items[candidate_id]
            intersection_size = len(query_items & candidate_items)
            if intersection_size > 0:
                score = intersection_size / len(candidate_items)
                scores.append((-score, candidate_id))

        # Get top-k results
        top_k_results = heapq.nsmallest(top_k, scores)
        return [(user_id, -score) for score, user_id in top_k_results]

    def process_user_batch(
        self, user_batch: List[str], top_k: int
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Process a batch of query users."""
        batch_results = {}
        for user_id in user_batch:
            batch_results[user_id] = self.find_similar_users(user_id, top_k)
        return batch_results

    def find_similar_users_parallel(
        self,
        query_users: List[str],
        top_k: int = 10,
        batch_size: int = 1000,
        n_workers: int = 4,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Find similar users for multiple query users in parallel."""
        # Split query users into batches
        batches = []
        for i in tqdm(range(0, len(query_users), batch_size), desc="Get batches"):
            batches.append(query_users[i : i + batch_size])
        results = {}

        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_user_batch, batch, top_k): batch
                for batch in batches
            }

            for future in tqdm(future_to_batch, desc="Processing batches"):
                batch_results = future.result()
                results.update(batch_results)

        return results


def main(args):
    if not os.path.exists(os.path.join(args.path, f"{args.category}.txt")):
        print(f"Preprocess start...")
        preprocess(args.category, ftype="jsonl", folder=args.path, save=True)
    else:
        print(f"{args.category}.txt exists")

    train, valid, test = data_partition(args.category, args.path)
    # history = pruning(train, args.min_length, args.max_length)
    users_list = list(train.keys())
    n_workers = cpu_count()  # Number of parallel processes

    # Initialize the finder
    print("Initializing UserSimilarityFinder...")
    finder = UserSimilarityFinder(train)

    print(f"Finding similar users for {len(users_list)} query users...")
    results = finder.find_similar_users_parallel(
        query_users=users_list,
        top_k=args.top_k,
        batch_size=args.batch_size,
        n_workers=n_workers,
    )

    # Save results
    print("Saving results...")
    with open(
        os.path.join(
            args.path,
            f"{args.category}_similar_users.json",
        ),
        "w",
    ) as f:
        json.dump(results, f)

    print("All completed !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get top-k similar users")
    parser.add_argument("--folder", type=str, default="/DATA/Recommendation/amazon")
    # parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--category", type=str, default="Books")
    parser.add_argument("--max_length", type=int, default=15)
    parser.add_argument("--min_length", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    args.path = os.path.join(args.folder, args.category)
    main(args)
