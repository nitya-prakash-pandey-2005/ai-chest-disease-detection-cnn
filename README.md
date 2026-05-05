# AI Chest Disease Detection (CNN)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 Project Overview

This repository contains a complete deep learning classification pipeline designed to detect 19 distinct medical anomalies (plus a "No Finding" healthy class) in chest X-ray images.

Developed originally for a competitive Kaggle environment, this project specifically addresses a severe evaluation constraint: a heavy **-5 penalty for false negatives** (missing a disease). Standard accuracy metrics were insufficient, requiring a highly customized approach to loss calculation and feature extraction.

## 🚀 Key Technical Achievements

- **High-Definition Architecture:** Upgraded standard 224x224 inputs to a **256x256 High-Definition pipeline**, utilizing Transfer Learning with `DenseNet121` to preserve faint, low-level medical features deep into the network.
- **Custom Loss Optimization:** Engineered a custom weighted Cross-Entropy loss function (5:1 ratio) to heavily penalize false negatives natively during backpropagation, bypassing brittle post-inference logit adjustments.
- **Hardware & Memory Management:** Successfully trained high-resolution tensors on constrained hardware (T4 GPU) by implementing dynamic batch sizing (reduced to 16) to prevent Out-Of-Memory (OOM) crashes.
- **Ensemble Experimentation:** Built and evaluated a mathematically optimized 75/25 soft-voting ensemble (DenseNet + EfficientNet) based on validation probabilities.

## 📊 Performance & Results

The final standalone HD DenseNet model outperformed the soft-voting ensemble, proving that higher resolution feature extraction was more valuable than multi-model blending for this specific dataset.

| Model Configuration        | Image Resolution | Hardware Constraint | Metric Score (Weighted F1) |
| :------------------------- | :--------------- | :------------------ | :------------------------- |
| ResNet50 (Baseline)        | 224x224          | Batch Size 32       | -4.72223                   |
| 75/25 Soft Ensemble        | 224x224          | Batch Size 32       | -4.44827                   |
| **DenseNet121 (Champion)** | **256x256**      | **Batch Size 16**   | **-4.44812 (Best)**        |

_(Note: The evaluation metric yields negative scores due to the extreme false-negative penalties. Values closer to zero indicate superior performance)._

## 📂 Repository Structure

```text
ai-chest-disease-detection-cnn/
│
├── main_pipeline.py     # Complete end-to-end training and inference script
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
