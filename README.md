<h1 align="center">
BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations 🔥
</h1>

<div align="center">

[![](https://img.shields.io/badge/BioT5-arxiv2310.07276-red?style=plastic&logo=GitBook)](https://arxiv.org/abs/2310.07276)
[![](https://img.shields.io/badge/BioT5+-arxiv2402.17810-red?style=plastic&logo=GitBook)](https://arxiv.org/abs/2402.17810)
[![](https://img.shields.io/badge/github-green?style=plastic&logo=github)](https://github.com/QizhiPei/BioT5) 
[![](https://img.shields.io/badge/model-pink?style=plastic&logo=themodelsresource)](https://huggingface.co/QizhiPei/biot5-base) 
[![](https://img.shields.io/badge/dataset-orange?style=plastic&logo=data.ai)](https://huggingface.co/datasets/QizhiPei/BioT5_finetune_dataset)
[![](https://img.shields.io/badge/Awesome-Biomolecule_Language_Cross_Modeling-orange?style=plastic&logo=awesomelists)](https://github.com/QizhiPei/Awesome-Biomolecule-Language-Cross-Modeling)
[![](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>


## News
🔥***July 11 2024***: *Codes, data, and pre-trained model for BioT5+ are relased.*

🔥***May 16 2024***: *[BioT5+](https://arxiv.org/abs/2402.17810) is accepted by ACL 2024 (Findings).*

🔥***Mar 03 2024***: *We have published a suvery paper [Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey](https://arxiv.org/abs/2403.01528) and the related github repository [Awesome-Biomolecule-Language-Cross-Modeling](https://github.com/QizhiPei/Awesome-Biomolecule-Language-Cross-Modeling). Kindly check it if you are interested in this field~*

🔥***Feb 29 2024***: *Update [BioT5](https://arxiv.org/abs/2310.07276) to [BioT5+](https://arxiv.org/abs/2402.17810) with the ability of IUPAC integration and multi-task learning!*

🔥***Nov 06 2023***: *Update [example usage](#example-usage) for molecule captioning, text-based molecule generation, drug-target interaction prediction!*

🔥***Oct 20 2023***: *The [data](#data) for fine-tuning is released!*

🔥***Oct 19 2023***: *The pre-trained and fine-tuned [models](#models) are released!*

🔥***Oct 11 2023***: *Initial commits. More codes, pre-trained model, and data are coming soon.*


## Overview
This repository contains the source code for 

* *EMNLP 2023* paper "[BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations](https://arxiv.org/abs/2310.07276)", by Qizhi Pei, Wei Zhang, Jinhua Zhu, Kehan Wu, Kaiyuan Gao, Lijun Wu, Yingce Xia, and Rui Yan. BioT5 achieves superior performance on various biological tasks.
* *ACL 2024 (Findings)* paper "[BioT5+: Towards Generalized Biological Understanding with IUPAC Integration and Multi-task Tuning](https://arxiv.org/abs/2402.17810)", by Qizhi Pei, Lijun Wu, Kaiyuan Gao, Xiaozhuan Liang, Yin Fang, Jinhua Zhu, Shufang Xie, Tao Qin, Rui Yan. BioT5+ is pre-trained and fine-tuned with a large number of experiments, including **3 types of problems (classification, regression, generation), 15 kinds of tasks, and 21 total benchmark datasets**, demonstrating the remarkable performance and state-of-the-art results in most cases.
* If you have questions, don't hesitate to open an issue or ask me via <qizhipei@ruc.edu.cn> or Lijun Wu via <lijun_wu@outlook.com>. We are happy to hear from you!

**↓Overview of BioT5**
![](./imgs/overview_biot5.png)

**↓Overview of BioT5+**
![](./imgs/overview_biot5+.png)

**Please refer to the `biot5` or `biot5_plus` folder for detailed instructions.**

## Citations
### BioT5
```
@inproceedings{pei2023biot5,
  title={BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations},
  author={Pei, Qizhi and Zhang, Wei and Zhu, Jinhua and Wu, Kehan and Gao, Kaiyuan and Wu, Lijun and Xia, Yingce and Yan, Rui},
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  month = dec,
  year = "2023",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.emnlp-main.70",
  pages = "1102--1123"
}
```
### BioT5+
```
@article{pei2024biot5+,
  title={BioT5+: Towards Generalized Biological Understanding with IUPAC Integration and Multi-task Tuning},
  author={Pei, Qizhi and Wu, Lijun and Gao, Kaiyuan and Liang, Xiaozhuan and Fang, Yin and Zhu, Jinhua and Xie, Shufang and Qin, Tao and Yan, Rui},
  journal={arXiv preprint arXiv:2402.17810},
  year={2024}
}
```

## Acknowledegments
The code is based on [nanoT5](https://github.com/PiotrNawrot/nanoT5).
