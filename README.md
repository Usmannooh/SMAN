# SMAN: Spatial-Channel Feature Enhancement with Multimodal Auxiliary Decoding Network for Radiology Report Generation

[![Conference](https://img.shields.io/badge/IEEE%20BIBM-2026-blue)](https://ieeebibm.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c)](https://pytorch.org/)
<!--
Official PyTorch implementation of **SMAN**, accepted at **IEEE BIBM 2026**.

> **SMAN: Spatial-Channel Feature Enhancement with Multimodal Auxiliary Decoding Network for Radiology Report Generation**
> Muhammad Usman, Chengbin Chen, Hongfei Lin, Zhang Yijia*
> *Dalian Maritime University & Dalian University of Technology, China*
> 📧 Correspondence: zhangyijia@dlmu.edu.cn

---

## 📰 Abstract

> Automated radiology report generation aims to produce clinically accurate textual descriptions from chest radiographs, reducing radiologists' workload while improving diagnostic consistency. Despite recent progress, existing methods still struggle to capture subtle pathological patterns and effectively integrate complementary visual representations. To address these limitations, we propose a Spatial-Channel Feature Enhancement with Multimodal Auxiliary Decoding Network (**SMAN**), a unified framework comprising four coordinated components: (1) a **Spatial-Channel Feature Enhancer (SCFE)** that jointly refines channel-wise pathology responses and spatial visual cues; (2) a **Cross-Representation Alignment and Relational Encoder (CARE)** that adaptively fuses local and global visual representations; (3) a training-only **M2Decoder** that provides auxiliary token-level supervision without additional inference overhead; and (4) **Global Feature-Norm Regularization (GFNR)** that stabilizes study-level visual representations under weak report-level supervision. Experiments on IU X-Ray and MIMIC-CXR demonstrate that SMAN achieves strong report generation performance on IU X-Ray and obtains the best BLEU scores together with the highest CheXbert-based clinical F1 on MIMIC-CXR.

---

##  Architecture

<p align="center">
  <img width="1253" height="586" alt="image" src="https://github.com/user-attachments/assets/35e9cdfb-2414-4c41-86a8-174574d4ae21" />

</p>

A ResNet-101 backbone processes multi-view chest radiographs; **SCFE** refines patch-level features, and **CARE** adaptively fuses local and global visual representations before the main Transformer report generator. During training only, **M2Decoder** provides auxiliary cross-modal supervision (CMI → CARE → TSA), and **GFNR** constrains pooled global feature magnitudes via an ℓ₂ norm penalty — both discarded at inference, so deployment cost is unchanged.

##  Key Contributions

- **SCFE** — jointly recalibrates channel-wise pathology responses and spatial localization cues inside the visual backbone, improving sensitivity to subtle chest X-ray abnormalities before global feature aggregation.
- **CARE** — a shared, reusable gating module that adaptively fuses patch-level and study-level global visual representations for more informative report generation.
- **M2Decoder** — a training-only auxiliary decoding branch that strengthens visual-textual grounding via token-level cross-modal supervision, with **zero inference-time cost**.
- **GFNR** — a simple ℓ₂ feature-norm penalty that stabilizes study-level visual embeddings under weak, report-level supervision, without requiring augmented views or positive/negative pairs.

---

##  Repository Structure

```
SMAN/
├── configs/                  # YAML configs for IU X-Ray / MIMIC-CXR experiments
├── data/
│   ├── iu_xray/               # IU X-Ray images + annotation json
│   └── mimic_cxr/             # MIMIC-CXR images + annotation json
├── models/
│   
│                    
│   ├── m2decoder.py             # Auxiliary cross-modal decoding branch                  
│   └── sman.py                   # Full SMAN model assembly
├── modules/
  └── RVFE.py
  └── ...
  └── trainer.py
├── maintrain.py
├── test.py
│

│   
├── requirements.txt
└── README.md
```


## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/SMAN.git
cd SMAN

conda create -n sman python=3.8 -y
conda activate sman

pip install -r requirements.txt
```

**Core dependencies**: `torch>=1.10`, `torchvision`, `transformers`, `pycocoevalcap`, `numpy`, `opencv-python`

---

## 📊 Datasets

| Dataset | Images | Reports | Source |
|---|---|---|---|
| **IU X-Ray** | 7,470 | 3,955 | [Open-i / IU Chest X-Ray](https://openi.nlm.nih.gov/faq) |
| **MIMIC-CXR** | 377,110 | 227,835 | [PhysioNet (credentialed access)](https://physionet.org/content/mimic-cxr/) |

Download the datasets, then preprocess:

```bash
python scripts/preprocess.py --dataset iu_xray --data_root data/iu_xray
python scripts/preprocess.py --dataset mimic_cxr --data_root data/mimic_cxr
```

---

##  Usage

**Training**

```bash
# IU X-Ray


# MIMIC-CXR

```

**Evaluation**



Key hyperparameters (see Section III of the paper): backbone learning rate `2×10⁻³` (StepLR, step=20, γ=0.5), other modules `7×10⁻⁴`, beam size `3` with trigram blocking, `λ_m2 = 5×10⁻⁵`, `λ_GFNR = 0.01`.

---

## 📈 Results

### NLG Metrics — IU X-Ray

| Model | BL-1 | BL-2 | BL-3 | BL-4 | MTR | RGL | CIDEr |
|---|---|---|---|---|---|---|---|
| R2Gen | 0.470 | 0.304 | 0.219 | 0.165 | 0.187 | 0.371 | – |
| R2GenCMN | 0.475 | 0.309 | 0.222 | 0.170 | 0.191 | 0.375 | – |
| **SMAN (Ours)** | **0.518** | **0.355** | **0.262** | **0.200** | **0.214** | **0.422** | **0.471** |

### NLG Metrics — MIMIC-CXR

| Model | BL-1 | BL-2 | BL-3 | BL-4 | MTR | RGL | CIDEr |
|---|---|---|---|---|---|---|---|
| R2Gen | 0.353 | 0.218 | 0.145 | 0.103 | 0.142 | 0.277 | – |
| R2GenCMN | 0.353 | 0.218 | 0.148 | 0.106 | 0.142 | 0.278 | – |
| **SMAN (Ours)** | **0.406** | **0.245** | **0.161** | **0.113** | 0.145 | 0.269 | 0.098 |

### Clinical Efficacy — MIMIC-CXR (CheXbert, micro-averaged)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| CmEAA | 0.505 | 0.330 | 0.399 |
| MPO | 0.436 | 0.376 | 0.353 |
| **SMAN (Ours)** | **0.519** | **0.387** | **0.476** |

Full comparisons, ablation studies (SCFE / CARE / TSA), and the GFNR sensitivity sweep are reported in the paper.

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{usman2026sman,
  title     = {{SMAN}: Spatial-Channel Feature Enhancement with Multimodal Auxiliary Decoding Network for Radiology Report Generation},
  author    = {Usman, Muhammad and Chen, Chengbin and Lin, Hongfei and Yijia, Zhang},
  booktitle = {2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026},
  publisher = {IEEE}
}
```

---

##  Acknowledgements

We thank the maintainers of **IU X-Ray** and **MIMIC-CXR** for making their datasets publicly available, and the authors of R2Gen, R2GenCMN, and related baselines whose open-source implementations supported our comparisons.


## 📬 Contact

For questions, please open a GitHub issue or email **zhangyijia@dlmu.edu.cn**.
-->
