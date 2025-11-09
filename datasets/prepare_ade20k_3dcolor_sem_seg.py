#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) HKUST-VGD Group
# Convert RGB images and RGB annotation maps with consistent ADE20K label colors, particularly for 3dcolor building dataset
# Usage:
#       change `TRAIN_NUM` and `SOURCE_DIR` accordingly
#       run `python prepare_ade20k_3dcolor_sem_seg.py`
# By Susan Yingshu Chen 25 Oct 2022

import glob
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
import shutil
import random
import argparse

ADE20K_SEM_SEG_CATEGORIES = [
    "other objects", #index = 0
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road, route",
    "bed",
    "window",
    "grass",
    "cabinet",
    "sidewalk, pavement",
    "person",
    "earth, ground",
    "door",
    "table",
    "mountain, mount",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting, picture",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock, stone",
    "wardrobe, closet, press",
    "lamp",
    "tub",
    "rail",
    "cushion",
    "base, pedestal, stand",
    "box",
    "column, pillar",
    "signboard, sign",
    "chest of drawers, chest, bureau, dresser",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator, icebox",
    "grandstand, covered stand",
    "path",
    "stairs",
    "runway",
    "case, display case, showcase, vitrine",
    "pool table, billiard table, snooker table",
    "pillow",
    "screen door, screen",
    "stairway, staircase",
    "river",
    "bridge, span",
    "bookcase",
    "blind, screen",
    "coffee table",
    "toilet, can, commode, crapper, pot, potty, stool, throne",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm, palm tree",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel, hut, hutch, shack, shanty",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning, sunshade, sunblind",
    "street lamp",
    "booth",
    "tv",
    "plane",
    "dirt track",
    "clothes",
    "pole",
    "land, ground, soil",
    "bannister, banister, balustrade, balusters, handrail",
    "escalator, moving staircase, moving stairway",
    "ottoman, pouf, pouffe, puff, hassock",
    "bottle",
    "buffet, counter, sideboard",
    "poster, posting, placard, notice, bill, card",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
    "canopy",
    "washer, automatic washer, washing machine",
    "plaything, toy",
    "pool",
    "stool",
    "barrel, cask",
    "basket, handbasket",
    "falls",
    "tent",
    "bag",
    "minibike, motorbike",
    "cradle",
    "oven",
    "ball",
    "food, solid food",
    "step, stair",
    "tank, storage tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket, cover",
    "sculpture",
    "hood, exhaust hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "trash can",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass, drinking glass",
    "clock",
    "flag",  # noqa
]

PALETTE = [
    (0, 0, 0),
    (120, 120, 120),
    (180, 120, 120),
    (6, 230, 230),
    (80, 50, 50),
    (4, 200, 3),
    (120, 120, 80),
    (140, 140, 140),
    (204, 5, 255),
    (230, 230, 230),
    (4, 250, 7),
    (224, 5, 255),
    (235, 255, 7),
    (150, 5, 61),
    (120, 120, 70),
    (8, 255, 51),
    (255, 6, 82),
    (143, 255, 140),
    (204, 255, 4),
    (255, 51, 7),
    (204, 70, 3),
    (0, 102, 200),
    (61, 230, 250),
    (255, 6, 51),
    (11, 102, 255),
    (255, 7, 71),
    (255, 9, 224),
    (9, 7, 230),
    (220, 220, 220),
    (255, 9, 92),
    (112, 9, 255),
    (8, 255, 214),
    (7, 255, 224),
    (255, 184, 6),
    (10, 255, 71),
    (255, 41, 10),
    (7, 255, 255),
    (224, 255, 8),
    (102, 8, 255),
    (255, 61, 6),
    (255, 194, 7),
    (255, 122, 8),
    (0, 255, 20),
    (255, 8, 41),
    (255, 5, 153),
    (6, 51, 255),
    (235, 12, 255),
    (160, 150, 20),
    (0, 163, 255),
    (140, 140, 200),
    (250, 10, 15),
    (20, 255, 0),
    (31, 255, 0),
    (255, 31, 0),
    (255, 224, 0),
    (153, 255, 0),
    (0, 0, 255),
    (255, 71, 0),
    (0, 235, 255),
    (0, 173, 255),
    (31, 0, 255),
    (11, 200, 200),
    (255, 82, 0),
    (0, 255, 245),
    (0, 61, 255),
    (0, 255, 112),
    (0, 255, 133),
    (255, 0, 0),
    (255, 163, 0),
    (255, 102, 0),
    (194, 255, 0),
    (0, 143, 255),
    (51, 255, 0),
    (0, 82, 255),
    (0, 255, 41),
    (0, 255, 173),
    (10, 0, 255),
    (173, 255, 0),
    (0, 255, 153),
    (255, 92, 0),
    (255, 0, 255),
    (255, 0, 245),
    (255, 0, 102),
    (255, 173, 0),
    (255, 0, 20),
    (255, 184, 184),
    (0, 31, 255),
    (0, 255, 61),
    (0, 71, 255),
    (255, 0, 204),
    (0, 255, 194),
    (0, 255, 82),
    (0, 10, 255),
    (0, 112, 255),
    (51, 0, 255),
    (0, 194, 255),
    (0, 122, 255),
    (0, 255, 163),
    (255, 153, 0),
    (0, 255, 10),
    (255, 112, 0),
    (143, 255, 0),
    (82, 0, 255),
    (163, 255, 0),
    (255, 235, 0),
    (8, 184, 170),
    (133, 0, 255),
    (0, 255, 92),
    (184, 0, 255),
    (255, 0, 31),
    (0, 184, 255),
    (0, 214, 255),
    (255, 0, 112),
    (92, 255, 0),
    (0, 224, 255),
    (112, 224, 255),
    (70, 184, 160),
    (163, 0, 255),
    (153, 0, 255),
    (71, 255, 0),
    (255, 0, 163),
    (255, 204, 0),
    (255, 0, 143),
    (0, 255, 235),
    (133, 255, 0),
    (255, 0, 235),
    (245, 0, 255),
    (255, 0, 122),
    (255, 245, 0),
    (10, 190, 212),
    (214, 255, 0),
    (0, 204, 255),
    (20, 0, 255),
    (255, 255, 0),
    (0, 153, 255),
    (0, 41, 255),
    (0, 255, 204),
    (41, 0, 255),
    (41, 255, 0),
    (173, 0, 255),
    (0, 245, 255),
    (71, 0, 255),
    (122, 0, 255),
    (0, 255, 184),
    (0, 92, 255),
    (184, 255, 0),
    (0, 133, 255),
    (255, 214, 0),
    (25, 194, 194),
    (102, 255, 0),
    (92, 0, 255),
]

_3D_COLOR_NAME = [
    "unlabeled,background",
    "building,edifice,statue",
    "car,automobile",
    "light,streetlight",
    "person,animal",
    "plant,flora",
    "road,ground,path,bridge,fence",
    "sky",
    "water,sea,lake", 
    "window,windowpane,door",
]
_3D_COLOR_TO_ADE20K_NAME = [
    "other objects",
    "building",
    "car",
    "light",
    "person",
    "plant",
    "road, route",
    "sky",
    "water", 
    "window",
]
_3D_COLOR_PALETTE = [
    (0, 0, 0),
    (180, 120, 120),
    (0,102,200),
    (255,173,0),
    (150,5,61),
    (204,255,4),
    (140,140,140),
    (6,230,230),
    (61, 230, 250), # water
    (230,230,230),
]

# assign ADE20K index to ours
_3D_COLOR_ADE20K_INDEX = np.zeros(len(_3D_COLOR_PALETTE)) 
for idx, color in enumerate(PALETTE):
    try:
        our_index = _3D_COLOR_PALETTE.index(color)
        _3D_COLOR_ADE20K_INDEX[our_index] = idx
    except:
        pass

# _3D_COLOR_ADE20K_INDEX =[]
# for name in _3D_COLOR_TO_ADE20K_NAME:
    # _3D_COLOR_ADE20K_INDEX.add(ADE20K_SEM_SEG_CATEGORIES.index(name))

# rgb to ade20k
def convert2ade20k(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    # print(img.shape) # HxWxC
    img_grayscale = np.zeros(img.shape[:2]).astype(np.uint8)
    for our_index, color in enumerate(_3D_COLOR_PALETTE):
        hw = np.bitwise_and(np.bitwise_and((img[:,:,0]==color[0]), (img[:,:,1]==color[1])), (img[:,:,2]==color[2]))
        # print(hw.shape)
        img_grayscale[hw] = _3D_COLOR_ADE20K_INDEX[our_index]
    Image.fromarray(img_grayscale).save(output)


# rgb to detectron2
def convert2d2(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    Image.fromarray(img).save(output)
# 
# TRAIN_NUM = 150
TRAIN_NUM = 1000
# TRAIN_RATIO = 0.8
SOURCE_DIR = "/home/susanchen/Desktop/cys_workspace/Data/3d_color/segmentation/Round2_2_Multi/Segmentation_R2_Multi-view"
if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "")) / "ADEChallengeData2016"

    source_dir = SOURCE_DIR
    source_image_dir = os.path.join(source_dir, "JPEGImages")
    source_segmentation_dir = os.path.join(source_dir, "SegmentationClass")
    source_segmentation_paths = sorted(glob.glob(os.path.join(source_segmentation_dir, "*")))
    # downloaded dataset from CVAT using segmentation1.1 format

    # Split training and validation set
    # Copy source images
    all_image_paths = sorted(glob.glob(os.path.join(source_image_dir, "*"))) # plz make sure there is only images
    # TRAIN_NUM = len(all_image_paths) * TRAIN_RATIO
    data_train_idx = random.sample(range(0, len(all_image_paths)), TRAIN_NUM)
    image_dir_train = os.path.join(dataset_dir, f"images/training/")   
    image_dir_val = os.path.join(dataset_dir, f"images/validation/") 
    segmentation_train_paths = []
    segmentation_val_paths = []
    os.makedirs(image_dir_train, exist_ok=True)
    os.makedirs(image_dir_val, exist_ok=True)
    for idx, image_path in enumerate(all_image_paths):
        assert Path(image_path).stem == Path(source_segmentation_paths[idx]).stem,\
        Path(image_path).stem + " & " + Path(source_segmentation_paths[idx]).stem

        if idx in data_train_idx:
            shutil.copy2(image_path, image_dir_train)
            segmentation_train_paths.append(source_segmentation_paths[idx])
        else:
            shutil.copy2(image_path, image_dir_val)
            segmentation_val_paths.append(source_segmentation_paths[idx])

    for name in ["training", "validation"]:
        image_dir = dataset_dir / "images/" / name
        annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "annotations_detectron2" / name
        annotation_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)

        # Convert RGB maps to index maps(ADE20K)
        print(f"Converting {name} RGB maps to ADE20K format.")
        if name == "training":
            for file_path in tqdm.tqdm(segmentation_train_paths):
                output_file = annotation_dir / Path(file_path).name
                # output_file = output_file.splitext(".")[:-1] + ".png"
                convert2ade20k(file_path, output_file)
        else:
            for file_path in tqdm.tqdm(segmentation_val_paths):
                output_file = annotation_dir / Path(file_path).name
                # output_file = output_file.splitext(".")[:-1] + ".png"
                convert2ade20k(file_path, output_file)

        # Convert annotation to detectron2 use
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert2d2(file, output_file)

        



