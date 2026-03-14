# Lung Cancer Detection using ResNet50

A deep learning system for **histopathological lung cancer classification** using **ResNet50 transfer learning**.

The model classifies microscopic tissue images into three categories:

- Lung Adenocarcinoma (`lung_aca`)
- Normal Lung Tissue (`lung_n`)
- Lung Squamous Cell Carcinoma (`lung_scc`)

This project includes:

- Training pipeline
- Model evaluation
- Confusion matrix and performance metrics
- Grad-CAM explainability
- FastAPI inference API

---

# Project Structure

```
lung-cancer-detection/
│
├── data/
│
├── notebooks/
│   └── lung_training.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
│
├── api/
│   └── main.py
│
├── outputs/
│   ├── models/
│   │   └── lung_model.keras
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── pr_curve.png
│
├── requirements.txt
└── README.md
```

---

# Dataset

Dataset used:

**Lung and Colon Cancer Histopathological Images**

Source:

https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

Only the **lung dataset** is used.

Classes:

| Class | Description |
|------|-------------|
| lung_aca | Lung Adenocarcinoma |
| lung_n | Normal Lung Tissue |
| lung_scc | Lung Squamous Cell Carcinoma |

---

# Model Architecture

Backbone:

```
ResNet50 (ImageNet pretrained)
```

Input size:

```
224 × 224 × 3
```

Classifier head:

```
GlobalAveragePooling2D
BatchNormalization
Dense(128, ReLU)
Dropout(0.4)
Dense(3, Softmax)
```

Training strategy:

- Transfer Learning
- Frozen convolutional base
- Custom classifier head

---

# Training Setup

Optimizer

```
Adam (learning rate = 1e-4)
```

Loss Function

```
Sparse Categorical Crossentropy
```

Regularization methods:

- Data Augmentation
- Dropout
- Batch Normalization
- Early Stopping
- ReduceLROnPlateau

---

# Model Performance

Validation results:

| Metric | Value |
|------|------|
Accuracy | **99.7%**
Balanced Accuracy | **99.7%**
Macro F1 Score | **0.997**
Matthews Correlation Coefficient | **0.995**

---

# Class-wise Metrics

| Class | Precision | Recall | F1 Score |
|------|------|------|------|
lung_aca | 0.997 | 0.997 | 0.997
lung_n | 1.000 | 1.000 | 1.000
lung_scc | 0.997 | 0.994 | 0.995

---

# Confusion Matrix

Total misclassifications:

```
9 errors out of ~3000 validation samples
```

Error rate:

```
≈ 0.3%
```

![Confusion Matrix](confusion_matrix.png)

---

# Training Behavior

Training accuracy

```
~98.9%
```

Validation accuracy

```
~99.7%
```

Generalization gap

```
~0.8%
```

This indicates **minimal overfitting and strong generalization**.

---

# Grad-CAM Explainability

Grad-CAM is used to visualize **which regions of histopathology images influence the model's predictions**.

Workflow:

```
Input Image
   ↓
Forward Pass
   ↓
Gradient Calculation
   ↓
Heatmap Generation
   ↓
Overlay on Original Image
```

This improves transparency and interpretability for medical AI systems.

---

# Running the Project

Install dependencies

```
pip install -r requirements.txt
```

---

## Train the model

```
python src/train.py
```

---

## Evaluate the model

```
python src/evaluate.py
```

---

## Run the inference API

```
uvicorn api.main:app --reload
```

Open API docs:

```
http://127.0.0.1:8000/docs
```

Upload an image and get predictions.

---

# Example API Response

```
{
  "prediction": "adenocarcinoma",
  "confidence": 0.997
}
```

---

# Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- FastAPI

---

# Author

Faizal  
