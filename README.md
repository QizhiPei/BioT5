<h1 align="center">
BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations 🔥
</h1>

<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://github.com/QizhiPei/BioT5)
[![](https://img.shields.io/badge/-github-green?style=plastic&logo=github)](https://github.com/QizhiPei/BioT5) 
</div>

## Overview
This repository contains the source code for *EMNLP 2023* paper "[BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations](https://github.com/QizhiPei/BioT5)", by Qizhi Pei, Wei Zhang, Jinhua Zhu, Kehan Wu, Kaiyuan Gao, Lijun Wu, Yingce Xia, and Rui Yan. BioT5 achieves superior performance on various biological tasks. If you have questions, don't hesitate to open an issue or ask me via <qizhipei@ruc.edu.cn> or Lijun Wu via <lijuwu@microsoft.com>. We are happy to hear from you!

![](./imgs/pipeline.png)

## News
**Oct 11 2023**: Initial commits. More codes, pre-trained model, and data are coming soon.

## Setup Environment
This is an example for how to set up a working conda environment to run the code.
```
git clone https://github.com/QizhiPei/BioT5.git
cd BioT5
conda create -n biot5 python=3.8
conda activate biot5
pip install -r requirements.txt
```

## Data
Coming soon...

## Pre-trained Models

|Model|Description|Huggingface Checkpoint|
|----|----|---|
|BioT5|Pre-trained BioT5||
|BioT5-Molecule Captioning|Fine-tuned BioT5 for molecule captioning task on ChEBI-20||
|BioT5-Text Based Molecule Generation|Fine-tuned BioT5 for text based molecule generation task on ChEBI-20||
|BioT5-PPI|Fine-tuned BioT5 for protein-protein interaction task on PEER benchmark||
|BioT5-DTI|Fine-tuned BioT5 for drug-target interaction task on PEER benchmark||
|BioT5-P|Fine-tuned BioT5 for protein property prediction task on PEER benchmark||


## Fine-tuning
Take the DTI task finetuning as an example:
```
bash finetune.sh ft_dti
```
Change the `ft_dti` to other file names in `biot5/configs/task` to fine-tune on other downstream tasks.


## About
### References
Coming soon...
```

```

### Acknowledegments
The code is based on [nanoT5](https://github.com/PiotrNawrot/nanoT5).