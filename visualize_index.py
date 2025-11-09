'''
Visualize indexed segmentation to colored maps
By Susan Yingshu Chen 22 Nov, 2022


usage:
python visualize_index.py grayscale_seg_dir rgb_seg_dir 

'''

import os
import glob
import argparse
import shutil
from PIL import Image,ImageChops,ImageOps
from PIL import ImageColor
import numpy as np
from tqdm import tqdm

ADE20K_SEM_SEG_CATEGORIES = [
	"other objects", #index = 0
	"wall",
	"building", # index = 2
	"sky", # index = 3
	"floor",
	"tree",
	"ceiling",
	"road, route", # index = 7
	"bed",
	"window",  # index = 9
	"grass",
	"cabinet",
	"sidewalk, pavement",
	"person", # index = 13
	"earth, ground",
	"door",
	"table",
	"mountain, mount",
	"plant", # index = 18
	"curtain",
	"chair",
	"car", # index = 21
	"water", # index = 22
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
	"light", # index = 83
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

_3D_COLOR_TO_ADE20K_NAME = [
	"other objects",
	"building",
	"sky",
	"road, route",
	"window",
	"person",
	"plant",
	"car",
	"water", 
	"light",
]

OUR_INDEX = set({}) # {0, 2, 3, 7, 9, 13, 18, 21, 22, 83}
for name in _3D_COLOR_TO_ADE20K_NAME:
	OUR_INDEX.add(ADE20K_SEM_SEG_CATEGORIES.index(name))

# print("Our indices: ")
# print(OUR_INDEX)

if __name__ == '__main__':
	parser = argparse.ArgumentParser() 
	parser.add_argument('index_image_folder', type=str, metavar='src_path', help='src_path')
	parser.add_argument('rgb_image_folder', type=str, metavar='out_path', help='out_path')
	opts = parser.parse_args()

	folders = sorted(os.listdir(opts.index_image_folder))
	# print(folders)

	for folder in folders:
		out_folder = os.path.join(opts.rgb_image_folder, folder)
		os.makedirs(out_folder, exist_ok=True)

		img_list = sorted(glob.glob(os.path.join(opts.index_image_folder, folder, "*"))) 
		
		for img_path in tqdm(img_list):      
			img = Image.open(img_path).convert("L") # in case RGBA format
			img = np.array(img)
			# INDEX = set({})            	
			out_img = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8) #W,H,3
			for i in OUR_INDEX:
				# if np.any(img == i):
					# INDEX.add(i+1)
				#### Visualization
				out_img[img == (i-1)] = PALETTE[i]

			# print("image %s has indices: "%(opts.index_image))
			# print(INDEX)
			# for idx in INDEX:
			# 	print(ADE20K_SEM_SEG_CATEGORIES[idx])


			result = Image.fromarray(out_img)
			filename = os.path.join(out_folder, os.path.basename(img_path))
			result.save(filename)

