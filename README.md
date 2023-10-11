# BioT5

This repository contains the code to reproduce the pre-training and fine-tuning of "BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations".
The code is based on [nanoT5](https://github.com/PiotrNawrot/nanoT5).

## Environment
You need to download this repository before running the following scripts.
```
cd biot5
conda create -n biot5 python=3.8
conda activate biot5
pip install -r requirements.txt
```

```
## Fine-tuning
Take the DTI task as an example:
```
bash finetune.sh ft_dti
```
Change the `ft_dti` to other file names in `BioT5/configs/task` to fine-tune on other downstream tasks.