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
  <img src="images/Artificial Neural Networks (ANN) Introduction.png" width=800 alt="MNIST Grid">
</p>

* **Description:** An introductory project to get familiar with PyTorch tensors, datasets, and foundational neural network concepts.
* **Dataset:** Loaded built-in via `torchvision.datasets.MNIST`.
* **Focus:** Learning basic forward propagation, loss functions, and optimization from scratch.

#### 02. Digit Recognition (CNN)

<p align="center">
  <img src="images/Srishti’s Hand - a handwritten typeface….png" width="600" alt="CNN Handwritten Digits Recognition Preview">
</p>

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
