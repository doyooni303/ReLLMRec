import os
import os.path
import gzip
import json
from tqdm import tqdm
from collections import defaultdict
from utils import save_SELFRec_data


def get_path(fname: str, ftype: str, folder: str):
    file_path = os.path.join(folder, f"{fname}.{ftype}")
    meta_path = os.path.join(folder, f"meta_{fname}.{ftype}")

    return file_path, meta_path


def parse(path):
    g = gzip.open(path, "rb")
    for l in tqdm(g):
        yield json.loads(l)


def open_json(file_path):
    with open(file_path, "r") as f:
        dict_list = [json.loads(line[:-1]) for line in f.readlines()]
    return dict_list


def preprocess(
    fname,
    ftype="jsonl",
    folder="/DATA/Recommendation/amazon/Books",
    save=False,
    data_list=None,
    meta_list=None,
):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    countI = defaultdict(lambda: 0)
    line = 0

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    # review_dict = {}
    # name_dict = {"title": {}, "description": {}}
    if fname == "Books":
        name_dict = {
            "title": {},
            "author": {},
            "average_rating": {},
            "features": {},
            "categories": {},
        }
    elif fname == "Movies_and_TV":
        name_dict = {
            "title": {},
            # "directors": {},
            "average_rating": {},
            # "description": {},
            "categories": {},
        }

    file_path, meta_path = get_path(fname, ftype, folder)

    if data_list is None:
        print(f"Loading jsonl files..")
        data_list = open_json(file_path)
    if meta_list is None:
        meta_list = open_json(meta_path)

    unique_data_list = []
    if ftype == "jsonl":
        # counting interactions for each user and item
        for l in tqdm(data_list):
            line += 1
            asin = l["asin"]
            pasin = l["parent_asin"]
            rev = l["user_id"]
            time = l["timestamp"]
            key = f"{asin}-{rev}-{time}"
            countI[key] += 1
            if countI[key] >= 2:
                continue
            else:
                countU[rev] += 1
                countP[asin] += 1
                unique_data_list.append(l)

        meta_dict = {}
        for l in meta_list:
            meta_dict[l["parent_asin"]] = l

        for l in tqdm(unique_data_list):
            line += 1
            asin = l["asin"]
            pasin = l["parent_asin"]
            rev = l["user_id"]
            time = l["timestamp"]

            threshold = 5
            if ("Beauty" in fname) or ("Toys" in fname):
                threshold = 4

            if countU[rev] < threshold or countP[asin] < threshold:
                continue

            if rev in usermap:
                userid = usermap[rev]
            else:
                usernum += 1
                userid = usernum
                usermap[rev] = userid
                User[userid] = []

            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid

            if [time, itemid] not in User[userid]:
                User[userid].append([time, itemid])

                for key in name_dict.keys():

                    if key in ["title", "average_rating"]:
                        name_dict[key][itemmap[asin]] = meta_dict[pasin].get(
                            key, "None"
                        )
                    elif key == "features":
                        features = meta_dict[pasin].get(key, "None")
                        if isinstance(features, list):
                            name_dict[key][itemmap[asin]] = " ".join(features)
                        else:
                            name_dict[key][itemmap[asin]] = features

                    elif key == "description":
                        try:
                            description = meta_dict[pasin].get(key, "None")
                            if isinstance(description, list):
                                if len(description) > 1:
                                    value = " ".join(description)[:1000]
                                else:
                                    value = description[0][:1000]
                            elif isinstance(description, str):
                                value = description[:1000]

                            name_dict[key][itemmap[asin]] = value
                        except:
                            print(f"Error: {asin}, description:{description}")
                            name_dict[key][itemmap[asin]] = "None"

                    elif key == "categories":
                        categories = meta_dict[pasin].get(key, "None")
                        if isinstance(categories, list):
                            name_dict[key][itemmap[asin]] = ", ".join(categories)
                        else:
                            name_dict[key][itemmap[asin]] = categories

                    elif key == "author":  # Books
                        try:
                            author_name = meta_dict[pasin][key].get("name", "Unknown")
                            author_info = meta_dict[pasin][key].get("about", "None")
                            if isinstance(author_info, list):
                                author_info = " ".join(author_info)
                            name_dict[key][
                                itemmap[asin]
                            ] = f"name: {author_name}, about: {author_info}"
                        except:
                            name_dict[key][itemmap[asin]] = "None"
                    elif key == "directors":  # Movies_and_TV
                        try:
                            directors = meta_dict[pasin]["details"]["Directors"]
                            if isinstance(directors, list):
                                name_dict[key][itemmap[asin]] = ", ".join(directors)
                            elif isinstance(directors, str):
                                name_dict[key][itemmap[asin]] = directors
                        except:
                            name_dict[key][itemmap[asin]] = "None"
            else:
                continue

    if save:
        json.dump(
            name_dict,
            open(os.path.join(folder, f"{fname}_meta_name_dict.json"), "w"),
        )
        print(f"Saved {fname}_meta_name_dict.json")
        for userid in User.keys():
            User[userid].sort(key=lambda x: x[0])

        print(usernum, itemnum)

        f = open(os.path.join(folder, f"{fname}.txt"), "w", encoding="UTF-8")
        for user in User.keys():
            for i in User[user]:
                f.write("%d %d\n" % (user, i[1]))
        f.close()
        print(f"Saved {fname}.txt")

        # usermap, itemmap save
        for map_, name in zip([usermap, itemmap], ["user", "item"]):
            with open(os.path.join(folder, f"{fname}_{name}map.json"), "w") as f:
                json.dump(map_, f)
            print(f"Saved {fname}_{name}map.json")

        save_SELFRec_data(fname, folder)
