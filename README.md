# 🩻 COVID-19 Detection from Chest X-Rays using DenseNet121

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Red?style=for-the-badge&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📌 Project Overview

This project implements a Deep Learning model to classify Chest X-Ray images into three distinct categories:
1.  **COVID-19**
2.  **Viral Pneumonia**
3.  **Normal**

Using **Transfer Learning** with the **DenseNet121** architecture, the model achieves exceptional sensitivity in detecting COVID-19 cases. A specialized **"Partial Unfreezing"** fine-tuning strategy was employed to adapt the pre-trained model to medical imaging features (such as ground-glass opacities) while preventing overfitting on a limited dataset.

🔗 **Kaggle Notebook:** [COVID-19 X-Ray Detection | DenseNet121](https://www.kaggle.com/code/goktani/covid-19-x-ray-detection-densenet121)

---

## 📂 Repository Structure

```text
COVID-19-XRay-Detection-DenseNet121/
│
├── COVID_19_XRay_Detection.ipynb   # Complete training & inference notebook
├── README.md                       # Project documentation
├── requirements.txt                # List of dependencies
├── images/                         # Visual results
│   ├── confusion_matrix.png        # Model performance matrix
│   ├── accuracy_loss_graph.png     # Training curves
│   └── gradcam_example.png         # Explainable AI (Heatmap) visualization
```
## 📊 Key Results
The model achieved an overall accuracy of 86% on the test set. Most importantly, it demonstrated perfect sensitivity for the critical COVID-19 class.

### Classification Report
Class,Precision,Recall,F1-Score,Support
Covid-19,1.00,0.96,0.98,26
Normal,0.76,0.80,0.78,20
Viral Pneumonia,0.80,0.80,0.80,20
Overall Accuracy,,,0.86,66

**Analysis:** The model successfully identified 25 out of 26 Covid-19 cases (Recall: 0.96) and had zero False Positives for Covid (Precision: 1.00). The confusion observed between Normal and Viral Pneumonia classes is attributed to the visual similarity of radiological features in mild pneumonia cases and the limited dataset size.

## 📈 Visual Performance
### 1. Training Performance
The model shows stable convergence without significant overfitting, thanks to the 2-stage training strategy.

### 2. Confusion Matrix
A detailed breakdown of predictions vs. ground truth. Note the high success rate in the first row (Covid).

### 3. Explainable AI: Grad-CAM
To ensure the model is medically relevant, I implemented Grad-CAM (Gradient-weighted Class Activation Mapping). The heatmap below shows the model focusing on the lung regions (opacity) rather than background artifacts to make a decision.

## 🧠 Methodology
### 1. Architecture: DenseNet121
DenseNet121 was chosen over deeper ResNet architectures because of its feature reuse mechanism. In medical imaging, low-level texture details are as important as high-level patterns. DenseNet connects each layer to every other layer in a feed-forward fashion, preserving these critical details.

### 2. Training Strategy: The "Partial Unfreezing" Technique
Instead of a standard fine-tuning approach, I used a two-stage process:

*Stage 1 (Warm-up): The base model was frozen. Only the custom classification head (GlobalAveragePooling -> BatchNormalization -> Dropout -> Dense) was trained.*

*Stage 2 (Partial Unfreezing): Instead of unfreezing the entire model (which risks catastrophic forgetting on small datasets), I unfroze only the last 50 layers. This allowed the model to adapt to X-Ray textures while keeping the foundational edge-detection filters intact.*

## 🚀 Installation & Usage
### 1. Clone the repository

```Bash
git clone [https://github.com/goktani/COVID-19-XRay-Detection-DenseNet121.git](https://github.com/goktani/COVID-19-XRay-Detection-DenseNet121.git)
cd COVID-19-XRay-Detection-DenseNet121
```
### 2. Install dependencies

```Bash
pip install -r requirements.txt
```

### 3. Run the Notebook Open COVID_19_XRay_Detection.ipynb in Jupyter Lab or VS Code to view the training process and run inference.

## 📬 Contact
Göktan İren - GitHub: @goktani
