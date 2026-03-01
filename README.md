# MORN: Hierarchical Multi-Omics Regulatory Network

Official PyTorch implementation of **MORN**, a hierarchical multi-omics regulatory network that models biologically grounded interactions between molecular profiles and histopathology images for cancer analysis.



---

## Overview

MORN is a multimodal graph learning framework designed to bridge molecular alterations and tissue morphology.

Unlike conventional multimodal fusion methods that only concatenate features, MORN explicitly models **cross-omics regulatory relationships** and aligns them with histopathology patterns through attention-based interaction.

The framework integrates four omics modalities:

* Copy Number Variation (CNV)
* DNA Methylation
* mRNA expression
* miRNA expression

together with whole-slide pathology images.

The model constructs a hierarchical heterogeneous graph linking molecular entities and patients, and performs graph message aggregation and cross-modal attention to obtain patient representations.

---

## Features

* Hierarchical multi-omics regulatory graph modeling
* Metapath-based contrastive representation learning
* Cross-modal interaction between omics and pathology
* Survival prediction and molecular subtype classification

---

## Installation

Create environment (example):

```bash
conda create -n morn python=3.10
conda activate morn

pip install torch torchvision torchaudio
pip install dgl
pip install numpy pandas scikit-learn
```

(Additional dependencies may be added later.)

---

## Training

Train the model using:

```bash
python main.py
```

---

## Data

This repository assumes preprocessed multi-omics features and extracted WSI features are prepared in advance.

Details will be released after acceptance.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{morn2026,
  title={A Hierarchical Multi-Omics Regulatory Network Linking Molecular Regulation to Histopathology},
  journal={MICCAI (under review)},
  year={2026}
}
```



