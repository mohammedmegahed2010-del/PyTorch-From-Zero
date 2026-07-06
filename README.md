# 🔥 PyTorch From Zero to Hero

<p align="center">
  <img src="images/Pytorch.png" width="100%" alt="PyTorch Logo">
</p>

A structured log of my journey into Deep Learning and Computer Vision. Progressing step-by-step from fundamental architectures to advanced models.

## 📊 Lab Projects & Progress

| Project Name | Architecture | Status |
| :--- | :--- | :---: |
| **01 MNIST Digit Classification** | Linear Model (ANN) | 🟢 Completed |
| **02 MNIST Digit Recognition** | Convolutional Network (CNN) | 🟢 Completed |
| **03 Medical Cost Prediction** | Linear Model (MLP) | 🟢 Completed |
| **04 Cats vs Dogs Image Classifier** | Transfer Learning (ResNet18) | 🟢 Completed |
| **05 Pneumonia X-Ray Detection** | Transfer Learning (ResNet18) | 🟢 Completed |
| **06 Brain Tumor Detection (MRI)** | Transfer Learning (ResNet34) | 🟡 In Progress |

## 🛠️ Tech Stack & Tools
<p align="left">
  <img src="images/Python_lobo.png" height="70" alt="Python">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="images/Pytorch_logo.png" height="70" alt="PyTorch">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="images/Kaggle_logo.png" height="70" alt="Kaggle">
</p>

## 📂 Structure
- Each project subfolder contains the implementation training notebook (`.ipynb`) and its standalone logic.
- 📦 **Model Weights:** All trained model weights (`.pth`) are stored permanently and can be downloaded from the [Releases Section](https://github.com/mohammedmegahed2010-del/PyTorch-From-Zero/releases/tag/v1.0).

## 📝 Projects Breakdown & Details

### 01. MNIST Digit Classification 🔢

<p align="center">
  <img src="https://data:image/png;base64,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" width="400" alt="MNIST Grid">
</p>

* **Description:** An introductory project to get familiar with PyTorch tensors, datasets, and foundational neural network concepts.
* **Dataset:** Loaded built-in via `torchvision.datasets.MNIST`.
* **Focus:** Learning basic forward propagation, loss functions, and optimization from scratch.

#### 02. Digit Recognition (CNN)
* **Description:** Transitioning from linear models to computer vision architectures to improve image recognition.
* **Architecture:** Custom Convolutional Neural Network (CNN).
* **Focus:** Understanding convolutional layers, pooling, and feature extraction.

#### 03. Medical Cost Prediction
* **Description:** A regression task aimed at predicting insurance medical costs based on tabular patient data.
* **Architecture:** Multi-Layer Perceptron (MLP).
* **Focus:** Handling continuous data, preprocessing tabular inputs, and evaluating regression metrics.

#### 04. Cats vs Dogs Image Classifier
* **Description:** A classic computer vision challenge to classify binary image datasets.
* **Architecture:** Transfer Learning with a pre-trained ResNet-18 baseline.
* **Focus:** Learning how to freeze early layers and modify the classification head for new visual tasks.

#### 05. Pneumonia X-Ray Detection
* **Description:** An advanced medical imaging application to detect pneumonia signs from chest X-Ray photos.
* **Architecture:** Fine-tuned Transfer Learning (ResNet-18).
* **Dataset:** Sourced from Kaggle medical chest X-Ray dataset (referenced in notebook comments).
* **Focus:** Applying deep learning skills to real-world clinical data and saving deployment-ready weights (`.pth`).
* **Metrics:** 83.20%

#### 06. Brain Tumor Detection (MRI)
* **Description:** A multi-class medical imaging application to classify brain MRI scans into four distinct categories (Healthy vs. 3 Tumor types).
* **Architecture:** Custom CNN / ResNet-34.
* **Focus:** Moving from binary to multi-class classification, handling grayscale MRI inputs, and preventing overfitting using advanced data augmentation.
