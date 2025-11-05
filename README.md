# City Segmentation
The segmentation model for ECCV 2024 paper [StyleCity]().

# Introduction
This repository is a duplicate repo of [Mask2Former](https://github.com/facebookresearch/Mask2Former), but we retrained the model with customized classes.

## Motivation
We found the existing pre-trained Mask2Former models have an unsatisfying performance in urban scenes, such as windows. Therefore, to fine-tune the model we follow the interactive 2D segmentation-and-refinement scheme and iteratively expand annotated data.

## Differences and Features
- Our model added more layers in the original network.
- Our model was pre-trained with ADE20K dataset, and iteratively finetuned with our customized city segmentation dataset with only **9 classes (esp. windows)**. <br>For more details, please refer to the paper [supplementary](https://www.chenyingshu.com/stylecity3d/assets/StyleCity_supp_doc.pdf) Sec.2.3 _Iterative 2D Segmentation Model Fine-tuning and Customization_.


# Quick Start

## Installation

### Example conda environment setup

```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone https://github.com/chenyingshu/city_segmentation.git
cd city_segmentation
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

For more details please refer to original repo's [installation instructions](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md).

### Pre-trained Checkpoint

TODO

## Inference 
Example usage to predict segmentation and get visualized segmenation map and indexed segmenation map.
```bash
cd demo
python demo.py
```

## Segmentation labels description
We following class indices used in ADE20K, but merge some classes into one, reducing classes into 9 in total.
|class | index |
| --- | --- |
| sky | |
| building | |
| window | |
| road | |
| plant | |
| light | |
| water | |
| car/vehicle | |
| person/animal | |

# Acknowledgement 
Code is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former).
