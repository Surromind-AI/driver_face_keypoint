import argparse
import ast
import glob
import json
import os
import tqdm
import numpy as np


def remove_extension(file_name):
    return file_name[:file_name.rfind(".")]


def get_name_map(path_lst):
    return {
        remove_extension(os.path.basename(path)): path
        for path in path_lst
    }


bbox_heuristic_usage = 0
def bbox_heuristic(label_path):
    global bbox_heuristic_usage
    bbox_heuristic_usage += 1
    min_x = min_y = 987654321
    max_x = max_y = 0
    with open(label_path) as f:
        for line in f.readlines():
            try:
                x, y = (float(x) for x in line.strip().split())
            except:
                continue
            min_x = min(x, min_x)
            max_x = max(x, max_x)
            min_y = min(y, min_y)
            max_y = max(y, max_y)
    w = max_x - min_x
    h = max_y - min_y
    return (
        min_x - w * 0.1,
        min_y - h * 0.1,
        max_x + w * 0.1,
        max_y + h * 0.1
    )


def split_data_list(lst, split_ratio):
    n = len(lst)
    split_total = sum(split_ratio)
    cumulative_split_ratio = [0]
    for s in split_ratio:
        cumulative_split_ratio.append(cumulative_split_ratio[-1] + s)
    split_point = tuple(c * n // split_total for c in cumulative_split_ratio)
    np.random.shuffle(lst)
    return [lst[left:right] for left, right in zip(split_point[:-1], split_point[1:])]


def split_m_f(lst):
    m_f = [x.split("_")[1] for x in lst]
    return [[lst[idx] for idx in range(len(lst)) if m_f[idx] == key] for key in ("M", "F")]


def convert_format(image_dir, label_dir, bbox_json, target_dir,
                   split_ratio):
    assert len(split_ratio) == 3
    image_map = get_name_map(glob.glob(os.path.join(image_dir, "*")))
    label_map = get_name_map(glob.glob(os.path.join(label_dir, "*")))
    matched_lst = list(set(image_map.keys()) & set(label_map.keys()))
    person_map = {}
    for file_name in matched_lst:
        file_name_split = file_name.split("_")
        idx = f"{file_name_split[1]}_{file_name_split[3]}"
        try:
            person_map[idx].append(file_name)
        except KeyError:
            person_map[idx] = [file_name]
    print(f"{len(image_map)} images, {len(label_map)} labels, {len(matched_lst)} matched")

    with open(bbox_json) as f:
        bbox = ast.literal_eval(f.read())

    data_list_split = [[] for _ in split_ratio]
    for filtered_person_lst in split_m_f(list(set(person_map.keys()))):
        for idx, person_lst_split in enumerate(split_data_list(filtered_person_lst, split_ratio)):
            for x in person_lst_split:
                data_list_split[idx].extend(person_map[x])
    
    for name_split, matched_lst_split in zip(("train", "valid", "test"), data_list_split):
        target_file = os.path.join(target_dir, f"{name_split}.lst")
        with open(target_file, mode="w") as f:
            for matched_name in tqdm.tqdm(matched_lst_split):
                image_path = image_map[matched_name]
                label_path = label_map[matched_name]
            
                image_name = os.path.basename(image_path)
                bbox_coord = bbox[image_name] if image_name in bbox else bbox_heuristic(label_path)
                print(image_path, label_path, " ".join(str(x) for x in bbox_coord), file=f)

    print(f"{bbox_heuristic_usage} missing bboxes filled with heuristic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="/data/keypoints/keypoints_images", type=str)
    parser.add_argument("--label-dir", default="/data/keypoints/keypoints_annotations", type=str)
    parser.add_argument("--bbox-json", default="/data/keypoints/keypoints_bbox/keypoints_bbox.json", type=str)
    parser.add_argument("--target-dir", default="/data/keypoints/converted", type=str)
    parser.add_argument("--split-train", default=8, type=int)
    parser.add_argument("--split-valid", default=1, type=int)
    parser.add_argument("--split-test", default=1, type=int)

    args = parser.parse_args()
    convert_format(args.image_dir, args.label_dir, args.bbox_json, args.target_dir,
                   (args.split_train, args.split_valid, args.split_test))
