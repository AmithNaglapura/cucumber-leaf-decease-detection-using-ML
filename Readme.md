# 🥒 Cucumber Leaf Disease Detection using VGG16

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Transfer%20Learning-purple?style=flat-square)

A deep learning-based image classification system for detecting diseases in cucumber leaves using **Transfer Learning with VGG16**. This project classifies cucumber leaf images into 5 categories — helping farmers and agricultural researchers identify plant health issues early.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Disease Classes](#disease-classes)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

This project uses a **pre-trained VGG16 convolutional neural network** (trained on ImageNet) with custom classification layers on top to detect 5 types of cucumber leaf conditions. The model leverages transfer learning to achieve high accuracy even on a relatively small dataset.

**Key Features:**
- Transfer learning with frozen VGG16 base
- Data augmentation to improve generalization
- Evaluation with confusion matrix, precision, recall, and F1-score
- 80/20 train-validation split
- Trained for 50 epochs with Adam optimizer

---

## 🌿 Disease Classes

| Class | Description |
|-------|-------------|
| **Anthracnose** | Fungal disease causing dark, sunken lesions on leaves |
| **Bacterial Wilt** | Bacterial infection causing rapid wilting of leaves and stems |
| **Downy Mildew** | Water mold causing yellow patches and grayish spores on leaves |
| **Fresh Leaf** | Healthy, disease-free cucumber leaf |
| **Gummy Stem Blight** | Fungal disease causing water-soaked lesions with gummy ooze |

---

## 📁 Project Structure

```
cucumber-leaf-disease-detection/
│
├── cucumber_disease_vgg16.py     # Main training and evaluation script
├── vgg16modelnew1epochs50.h5     # Saved trained model (generated after training)
├── README.md                     # Project documentation
│
└── cucumber70/                   # Dataset directory
    ├── Anthracnose/
    ├── Bacterial Wilt/
    ├── Downy Mildew/
    ├── Fresh Leaf/
    └── Gummy Stem Blight/
```

---

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Seaborn
- Pillow (PIL)
- scikit-learn

---

## ⚙️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/cucumber-leaf-disease-detection.git
cd cucumber-leaf-disease-detection
```

**2. Create and activate a virtual environment (recommended):**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**3. Install the required packages:**
```bash
pip install tensorflow numpy matplotlib seaborn pillow scikit-learn
```

---

## 📂 Dataset Setup

1. Organize your dataset in the following structure:
```
cucumber70/
├── Anthracnose/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Bacterial Wilt/
├── Downy Mildew/
├── Fresh Leaf/
└── Gummy Stem Blight/
```

2. Update the `dataset_dir` path in the script to point to your local dataset:
```python
dataset_dir = 'path/to/your/cucumber70'
```

> **Note:** The dataset used in this project contains 70 images per class. For better performance, more images per class are recommended.

---

## 🚀 Usage

Run the main script to train and evaluate the model:

```bash
python cucumber_disease_vgg16.py
```

**What the script does:**

1. **Displays sample images** — Shows 3 random images from each disease class
2. **Prepares data** — Applies augmentation (shear, zoom, horizontal flip) and splits into train/validation sets
3. **Builds the model** — Loads pretrained VGG16 and adds custom Dense layers
4. **Trains the model** — Trains for 50 epochs and saves the model as `vgg16modelnew1epochs50.h5`
5. **Evaluates performance** — Reports accuracy, loss, precision, recall, and F1-score
6. **Plots results** — Generates training/validation accuracy curves and a confusion matrix

---

## 🧠 Model Architecture

```
VGG16 (pretrained on ImageNet, layers frozen)
    ↓
Flatten
    ↓
Dense(128, activation='relu')
    ↓
Dense(64, activation='relu')
    ↓
Dense(5, activation='softmax')   ← 5 disease classes
```

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Input Shape | 224 × 224 × 3 |
| Batch Size | 32 |
| Epochs | 50 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Validation Split | 20% |
| Base Model | VGG16 (frozen) |

---

## 📊 Results

After training, the script outputs:

- **Training & Validation Accuracy curve** — Visualizes model learning over epochs
- **Validation Accuracy & Loss** — Final evaluation metrics on the validation set
- **Precision, Recall, F1-Score** — Weighted metrics across all 5 classes
- **Confusion Matrix** — Heatmap showing true vs. predicted class distribution

> Results will vary depending on your dataset size and hardware. For best results, consider using a GPU and a larger dataset.

---

## 🔧 Customization

You can easily adjust the following parameters in the script:

```python
batch_size = 32          # Increase if you have more GPU memory
epochs = 50              # Increase for potentially better accuracy
image_shape = (224, 224, 3)  # Fixed for VGG16 input
validation_split = 0.2   # Adjust train/val ratio as needed
```

To **unfreeze VGG16 layers** for fine-tuning (advanced):
```python
for layer in base_model.layers[-4:]:  # Unfreeze last 4 layers
    layer.trainable = True
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [VGG16 Paper](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman, 2014
- [TensorFlow / Keras](https://www.tensorflow.org/) — Deep learning framework
- [ImageNet](https://www.image-net.org/) — Pretrained weights source

---

<p align="center">Made with ❤️ for smarter agriculture</p>
