# ClearSCD

## Introduction
A remote sensing semantic change detection model, Comprehensively leveraged sEmantics and chAnge Relationships Semantics Change Detection model, named ClearSCD.

This new method draws inspiration from the mutual reinforcement of semantic and change information in the multi-task learning model. 

![](figs/ClearSCD.png)
Overview of the ClearSCD.

## Innovations
The main innovations in ClearSCD are as follows:<br>
Firstly, we introduced a supervised Semantics Augmented Contrastive Learning (SACL) module, utilizing both local and global data features, along with cross-temporal differences. 

Secondly, a Bi-temporal Semantic Correlation Capture (BSCC) mechanism is designed, allowing for the refinement of semantics through the output of the Binary Change Detection (BCD) branch.

Lastly, a deep CVAPS module in classification posterior probability space is developed to execute BCD by integrating semantics posterior probabilities instead of high-dimensional features.


## Getting Started
1. Download [Hi-UCD series dataset](https://github.com/Daisy-7/Hi-UCD-S).
2. Deal with the dataset using clip_image.py, deal_hiucd.py and write_path.py from the floder scripts.<br>
   **Note:** After running the deal_hiucd.py,  the classification codes in Hi-UCD with the land cover class in order minus 1, unlabeled region as 9 in bi-temporal semantic maps, and unlabelled as 255 in BCD.
4. Run main.py, then you will find the checkpoints in the results folder.
