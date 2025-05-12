import os
import random
from collections import defaultdict
from tqdm import tqdm
from typing import List


def write_text_file(file_path: str, data: List):
    with open(file_path, "w") as f:
        f.writelines(data)


# train/val/test data generation
def data_partition(fname, path=None, min_length=5):
    usernum = 0
    itemnum = 0

    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    # assume user/item index starting from 1

    f = open(os.path.join(path, f"{fname}.txt"), "r")

    for line in f:
        u, i = line.rstrip().split(" ")
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in tqdm(User, desc="splitting data by user"):
        nfeedback = len(User[user])
        # if nfeedback < min_length:
        #     user_train[user] = User[user]
        #     user_valid[user] = []
        #     user_test[user] = []

        # else:
        if nfeedback >= min_length:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return [user_train, user_valid, user_test]


def save_SELFRec_data(
    fname, path: str = None, min_length: int = 5, max_length: int = 15
):
    train, valid, test = data_partition(fname, path, min_length)
    selfrec_train, selfrec_test = [], []

    for user in tqdm(train.keys()):
        v = train[user][-max_length:]
        v.append(valid[user][-1])
        selfrec_train.append(f"{user}:{' '.join([str(i) for i in v[:-1]])}" + "\n")
        selfrec_test.append(f"{user}:{test[user][-1]}\n")

    write_text_file(os.path.join(path, f"train.txt"), selfrec_train)
    write_text_file(os.path.join(path, f"test.txt"), selfrec_test)

    print(f"Saved SELFRec data to {path}")
