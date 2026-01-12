## The ClearSCD model: Comprehensively leveraging semantics and change relationships for semantic change detection in high spatial resolution remote sensing imagery
[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271624001734?via%3Dihub)

## Introduction
A remote sensing semantic change detection model, Comprehensively leveraged sEmantics and chAnge Relationships Semantics Change Detection model, named ClearSCD.

This new method draws inspiration from the mutual reinforcement of semantic and change information in the multi-task learning model. 

![](figs/ClearSCD.png)
Overview of the ClearSCD.

## Innovations
The main innovations in ClearSCD are as follows:

1. We introduced a supervised Semantics Augmented Contrastive Learning (SACL) module, utilizing both local and global data features, along with cross-temporal differences. 

2. A Bi-temporal Semantic Correlation Capture (BSCC) mechanism is designed, allowing for the refinement of semantics through the output of the Binary Change Detection (BCD) branch.

3. A deep CVAPS module in classification posterior probability space is developed to execute BCD by integrating semantics posterior probabilities instead of high-dimensional features.

## Requirements
1. The pytorch version of torchvision>=0.13.1 is recommended to ensure that the torchvision library contains Efficientnet's pretrained weights.
2. Then `pip install segmentation-models-pytorch` to install a Python library [Segmentation Models Pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) for image segmentation based on PyTorch.

## Getting Started
1. Download [Hi-UCD series dataset](https://github.com/Daisy-7/Hi-UCD-S).
   
2. Deal with the dataset using `clip_image.py`, `deal_hiucd.py`, and `write_path.py` from the folder scripts.<br>
   **Note: After running the `deal_hiucd.py`,  the classification codes in Hi-UCD with the land cover class in order minus 1, the unlabeled region as 9 in bi-temporal semantic maps, and unlabeled as 255 in BCD.**
   
3. Run `main.py`, then you will find the checkpoints in the results folder.

4. Run `test.py`, then you will obtain the test metric and visual results. Our checkpoint on the Hi-UCD-mini dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/13U_luASmmVsrQNEK2SPrJRCEfQROWf8p/view?usp=sharing)

## Citation
If you use the ClearSCD codes or the LsSCD dataset, please cite our paper:
```bibtex
@article{tang2024clearscd,
title = {The ClearSCD model: Comprehensively leveraging semantics and change relationships for semantic change detection in high spatial resolution remote sensing imagery},
author = {Tang, Kai and Xu, Fei and Chen, Xuehong and Dong, Qi and Yuan, Yuheng and Chen, Jin},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {211},
pages = {299-317},
year = {2024},
issn = {0924-2716},
}
```

## Future
We will publish a large-scale semantic change detection (LsSCD) dataset, which consists of Google Earth images from September 2013 and August 2015, with a spatial resolution of 0.6 m and a full size of 48000 × 32500 pixels.

LsSCD reveals urban and rural land cover changes in the city of Nanjing, the capital of Jiangsu Province, China. 

Seven LULC types, including building, road, water, bare land, tree, cropland, and others, were recorded in LsSCD.

[LsSCD download link](http://www.chen-lab.club/?page_id=11432) (comming soon)

![](figs/LsSCD.png)
Overview of the LsSCD.
