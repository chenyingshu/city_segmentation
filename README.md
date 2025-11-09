# City Segmentation
The 2D segmentation model for ECCV 2024 paper [StyleCity](ttps://www.chenyingshu.com/stylecity3d/assets/StyleCity).

# Introduction
This repository is a duplicate repo of [Mask2Former](https://github.com/facebookresearch/Mask2Former), but we retrained the model with customized classes.

## Motivation
We found the existing pre-trained Mask2Former models have an unsatisfying performance in urban scenes, such as windows. Therefore, to fine-tune the model we follow the interactive 2D segmentation-and-refinement scheme and iteratively expand annotated data.

## Differences and Features
- Our model added more layers in the original network.
- Our model was pre-trained with ADE20K dataset (continue training from swin_large_patch4_window12_384_22k.pkl), and iteratively finetuned with our customized city segmentation dataset with only **9 classes (esp. windows)**. <br>For more details, please refer to the paper [supplementary](https://www.chenyingshu.com/stylecity3d/assets/StyleCity_supp_doc.pdf) Sec.2.3 _Iterative 2D Segmentation Model Fine-tuning and Customization_.


# Quick Start

## Installation

### Example conda environment setup

```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone https://github.com/chenyingshu/city_segmentation.git
cd city_segmentation
# Note that directly use modified detectron2 in this repo
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
cd ..
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

For more details please refer to original repo's [installation instructions](INSTALL.md).
But note that please use the modified detectron2 in this repo.

### Pre-trained Checkpoint

You can download out pre-trained checkpoint [here](https://hkust-vgd.ust.hk/stylecity3d/checkpoints/city_segmentation/model_final_v3.pth), and put it under the folder `checkpoint`.

## Inference 
Example usage to predict 2D segmentation and get visualized segmenation map and indexed segmenation map.
```bash
cd demo
python demo.py \
--config-file ../configs/ade20k/semantic-segmentation/swin/anh_maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml \
--input ../images \
--output ../results \
--opts MODEL.WEIGHTS ../checkpoint/model_final_v3.pth
```

For 3D segmentation usage, please refer to [StyleCity repo](https://github.com/chenyingshu/stylecity3d).

## Segmentation labels description
We following class indices used in ADE20K, but merge some classes into one, reducing the number of classes into 10 in total, including 9 classes of interests. Note that in `detectron2`, index should be index+1, and 0 to be 255.

| class name | ADE20K name | index |
| --- | --- | --- |
| unlabeled,background | other objects | 0 |
| building,edifice,statue | building | 2|
| sky | sky | 3 |
| road,ground,path,bridge,fence | road,route | 7 | 
| window,windowpane,door | window | 9 |
| person,animal | person | 13 |
| plant,flora | plant | 18 |
| car,automobile,vehicle | car | 21 | 
| water,sea,lake | water | 22 |
| light,streetlight | light | 83 |

# More Inference and Training Usage
Please refer to [GETTING_STARTED](GETTING_STARTED.md) for more details.

# Acknowledgement 
We are grateful for the contribution of the work "Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)". Our code is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former).
