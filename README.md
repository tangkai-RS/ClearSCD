# ClearSCD

## Introduction
A remote sensing semantic change detection model, Comprehensively leveraged sEmantics and chAnge Relationships Semantics Change Detection model, named ClearSCD.

This new method draws inspiration from the mutual reinforcement of semantic and change information in the multi-task learning model. 

## Innovations
The main innovations in ClearSCD are as follows: 
Firstly, we introduced a supervised Semantics Augmented Contrastive Learning (SACL) module, utilizing both local and global data features, along with cross-temporal differences. 

Secondly, a Bi-temporal Semantic Correlation Capture (BSCC) mechanism is designed, allowing for the refinement of semantics through the output of the Binary Change Detection (BCD) branch.

Lastly, a deep CVAPS module in classification posterior probability space is developed to execute BCD by integrating semantics posterior probabilities instead of high-dimensional features.
