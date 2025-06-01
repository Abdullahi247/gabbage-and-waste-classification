# 🗑️ Garbage Classification – Recyclable Waste Image Classifier

This project is a deep learning-based image classification system that automatically categorizes waste images into 12 garbage classes. It is designed to support smart waste management and environmental sustainability by enabling intelligent sorting of recyclable vs non-recyclable materials.

Built with PyTorch and a ResNet18 backbone, the model achieves strong accuracy and generalization across diverse categories.

---

## 📌 Project Objectives

- Automatically classify garbage images into one of 12 distinct categories.
- Improve accuracy in sorting recyclable materials using image-based recognition.
- Provide a clean, reproducible training and prediction pipeline.
- Serve as a foundation for future work in **object detection** and **smart waste bins**.

---

## Data Source:

-The dataset used to train this model was sourced from https://www.kaggle.com/datasets/mostafaabla/garbage-classification

## 🗂️ Dataset Classes

The model classifies images into the following categories:

---

## 📈 Model Performance

| Metric           | Value |
|------------------|-------|
| Accuracy         | 88%   |
| Macro F1-score   | 0.85  |
| Weighted F1-score| 0.89  |

**Sample Per-Class Performance:**


---

---

## 🧠 Model Details

- **Architecture**: ResNet18 (pretrained)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
- **Training Epochs**: 10
- **Input Size**: 224x224 RGB
- **Framework**: PyTorch


---

## 🚀 Getting Started

###  Clone the Repository


git clone https://github.com/Abdullahi247/gabbage-and-waste-classification.git

python main.py

python predict.py

You’ll get a prediction like

🧠 Predicted class: green-glass
