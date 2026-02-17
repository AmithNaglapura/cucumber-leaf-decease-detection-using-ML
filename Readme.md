<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4CAF50,100:2E7D32&height=200&section=header&text=Cucumber%20Leaf%20Disease%20Detection&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Deep%20Learning%20%7C%20VGG16%20Transfer%20Learning%20%7C%20TensorFlow&descAlignY=58&descSize=16" width="100%"/>

<br/>

<!-- BADGES ROW 1 -->
<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>

<br/><br/>

<!-- BADGES ROW 2 -->
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Model-VGG16-8B5CF6?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Classes-5-EC4899?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Epochs-50-F59E0B?style=for-the-badge"/>

<br/><br/>

<p align="center">
  <b>🌱 An intelligent plant health monitoring system that detects cucumber leaf diseases<br/>using state-of-the-art Transfer Learning with VGG16.</b>
</p>

<br/>

</div>

---

## 📋 Table of Contents

<details>
<summary>Click to expand</summary>

- [✨ Overview](#-overview)
- [🌿 Disease Classes](#-disease-classes)
- [⚡ Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [📦 Requirements](#-requirements)
- [⚙️ Installation](#️-installation)
- [📂 Dataset Setup](#-dataset-setup)
- [🚀 Usage](#-usage)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Evaluation & Results](#-evaluation--results)
- [🔧 Customization & Fine-Tuning](#-customization--fine-tuning)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

</details>

---

## ✨ Overview

<table>
<tr>
<td>

This project builds a **high-accuracy plant disease classifier** using **Transfer Learning** on top of the renowned VGG16 architecture. By freezing the pretrained convolutional layers and training only the custom classification head, the model achieves strong performance even with limited agricultural image data.

**Why this matters:**
> Early detection of cucumber diseases can prevent up to **70% of crop losses**, offering a practical AI-powered tool for farmers, agronomists, and precision agriculture applications.

</td>
</tr>
</table>

### 🔑 Key Features

| Feature | Details |
|---------|---------|
| 🏗️ **Architecture** | VGG16 pretrained on ImageNet + custom classifier head |
| 🔒 **Transfer Learning** | Base layers frozen — only top layers trained |
| 🔄 **Data Augmentation** | Shear, zoom, and horizontal flip to improve robustness |
| 📐 **Train/Val Split** | 80% Training / 20% Validation |
| 📈 **Metrics** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| 💾 **Model Export** | Saves trained model as `.h5` for reuse or deployment |

---

## 🌿 Disease Classes

The model detects **5 cucumber leaf conditions**:

<table>
<thead>
<tr>
<th align="center">🔬 Class</th>
<th align="center">🏷️ Type</th>
<th>📝 Description</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><b>🟤 Anthracnose</b></td>
<td align="center"><code>Fungal</code></td>
<td>Causes dark, sunken circular lesions on leaves, stems, and fruits; thrives in humid conditions</td>
</tr>
<tr>
<td align="center"><b>💧 Bacterial Wilt</b></td>
<td align="center"><code>Bacterial</code></td>
<td>Rapid wilting of leaves and vines caused by <em>Erwinia tracheiphila</em>, spread by cucumber beetles</td>
</tr>
<tr>
<td align="center"><b>🟡 Downy Mildew</b></td>
<td align="center"><code>Oomycete</code></td>
<td>Produces yellow angular patches on upper leaf surface with grayish-purple spores beneath</td>
</tr>
<tr>
<td align="center"><b>🟢 Fresh Leaf</b></td>
<td align="center"><code>Healthy</code></td>
<td>Normal, disease-free cucumber leaf with vibrant green color and intact surface structure</td>
</tr>
<tr>
<td align="center"><b>⚫ Gummy Stem Blight</b></td>
<td align="center"><code>Fungal</code></td>
<td>Water-soaked lesions that ooze amber gummy substance; affects leaves, stems, and crowns</td>
</tr>
</tbody>
</table>

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/cucumber-leaf-disease-detection.git
cd cucumber-leaf-disease-detection

# 2. Install dependencies
pip install tensorflow numpy matplotlib seaborn pillow scikit-learn

# 3. Set your dataset path in the script, then run
python cucumber_disease_vgg16.py
```

---

## 📁 Project Structure

```
📦 cucumber-leaf-disease-detection/
│
├── 📄 cucumber_disease_vgg16.py       ← Main training & evaluation script
├── 🤖 vgg16modelnew1epochs50.h5       ← Saved model (auto-generated after training)
├── 📘 README.md                       ← Project documentation
│
└── 📂 cucumber70/                     ← Dataset root directory
    ├── 🟤 Anthracnose/
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    ├── 💧 Bacterial Wilt/
    ├── 🟡 Downy Mildew/
    ├── 🟢 Fresh Leaf/
    └── ⚫ Gummy Stem Blight/
```

---

## 📦 Requirements

<table>
<tr>
<td>

**Core Libraries**
- `tensorflow >= 2.0`
- `numpy`
- `matplotlib`
- `seaborn`

</td>
<td>

**Supporting Libraries**
- `Pillow (PIL)`
- `scikit-learn`
- `keras` *(bundled with TF)*

</td>
<td>

**Hardware (Recommended)**
- GPU with CUDA support
- 8 GB+ RAM
- 4 GB+ VRAM

</td>
</tr>
</table>

---

## ⚙️ Installation

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/cucumber-leaf-disease-detection.git
cd cucumber-leaf-disease-detection
```

### Step 2 — Create a Virtual Environment *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install tensorflow numpy matplotlib seaborn pillow scikit-learn
```

> 💡 **Tip:** For GPU support, install `tensorflow-gpu` instead of `tensorflow` and ensure your CUDA/cuDNN versions are compatible.

---

## 📂 Dataset Setup

### Folder Structure
Organize your images into class-named subdirectories:

```
cucumber70/
├── Anthracnose/        ← ~70 images
├── Bacterial Wilt/     ← ~70 images
├── Downy Mildew/       ← ~70 images
├── Fresh Leaf/         ← ~70 images
└── Gummy Stem Blight/  ← ~70 images
```

### Update the Path
Open `cucumber_disease_vgg16.py` and set your local path:

```python
dataset_dir = 'path/to/your/cucumber70'   # ← Update this line
```

> ⚠️ **Important:** The default dataset contains ~70 images per class. For production-level accuracy, a dataset of **500+ images per class** is strongly recommended.

---

## 🚀 Usage

Run the full pipeline with a single command:

```bash
python cucumber_disease_vgg16.py
```

### What Happens Step-by-Step

```
Step 1 ── 🖼️  Visualizes 3 sample images per disease class
    │
Step 2 ── 🔄  Applies data augmentation & creates train/val generators
    │
Step 3 ── 🏗️  Loads VGG16 base + builds custom classification head
    │
Step 4 ── 🏋️  Trains the model for 50 epochs (saves to .h5)
    │
Step 5 ── 📊  Evaluates on validation set (accuracy, loss)
    │
Step 6 ── 📈  Plots accuracy curves & confusion matrix heatmap
    │
Step 7 ── 🎯  Prints Precision, Recall, and F1-Score
```

---

## 🧠 Model Architecture

### Network Diagram

```
┌─────────────────────────────────────────────────┐
│            INPUT IMAGE  224 × 224 × 3            │
└────────────────────────┬────────────────────────┘
                         │
┌────────────────────────▼────────────────────────┐
│                     VGG16                        │
│          (Pretrained on ImageNet)                 │
│              ❄️  Layers Frozen                   │
│   Conv → Pool → Conv → Pool → Conv → Pool ...   │
└────────────────────────┬────────────────────────┘
                         │
               ┌─────────▼─────────┐
               │      Flatten       │
               └─────────┬─────────┘
                         │
               ┌─────────▼─────────┐
               │   Dense(128, ReLU) │
               └─────────┬─────────┘
                         │
               ┌─────────▼─────────┐
               │   Dense(64, ReLU)  │
               └─────────┬─────────┘
                         │
               ┌─────────▼─────────┐
               │ Dense(5, Softmax)  │  ← 5 Disease Classes
               └───────────────────┘
```

### Training Configuration

<table>
<thead>
<tr>
<th>⚙️ Parameter</th>
<th>📌 Value</th>
<th>💬 Notes</th>
</tr>
</thead>
<tbody>
<tr><td><b>Input Shape</b></td><td><code>224 × 224 × 3</code></td><td>Standard VGG16 input size</td></tr>
<tr><td><b>Batch Size</b></td><td><code>32</code></td><td>Balanced for memory and speed</td></tr>
<tr><td><b>Epochs</b></td><td><code>50</code></td><td>Adjustable based on convergence</td></tr>
<tr><td><b>Optimizer</b></td><td><code>Adam</code></td><td>Adaptive learning rate</td></tr>
<tr><td><b>Loss Function</b></td><td><code>Categorical Crossentropy</code></td><td>Multi-class classification</td></tr>
<tr><td><b>Validation Split</b></td><td><code>20%</code></td><td>Stratified random split</td></tr>
<tr><td><b>Base Model</b></td><td><code>VGG16 (frozen)</code></td><td>ImageNet pretrained weights</td></tr>
<tr><td><b>Output Activation</b></td><td><code>Softmax</code></td><td>Probability over 5 classes</td></tr>
</tbody>
</table>

---

## 📊 Evaluation & Results

After training completes, the following outputs are generated:

### 📈 Accuracy Curves
A side-by-side plot of **Training vs. Validation Accuracy** across all 50 epochs — useful for detecting overfitting or underfitting.

### 🔢 Quantitative Metrics
```
Validation Accuracy : XX.XX%
Validation Loss     : X.XXXX
Precision (weighted): X.XXXX
Recall (weighted)   : X.XXXX
F1 Score (weighted) : X.XXXX
```

### 🗺️ Confusion Matrix
A color-coded heatmap (Blues palette) showing how well the model predicts each disease class — reveals which diseases are commonly confused with each other.

> 💡 **Note:** Actual metric values depend on your dataset and hardware. GPU training with a larger dataset yields significantly better results.

---

## 🔧 Customization & Fine-Tuning

### Adjust Hyperparameters
```python
batch_size       = 32              # ↑ Increase if GPU memory allows
epochs           = 50              # ↑ More epochs = potentially better accuracy
image_shape      = (224, 224, 3)   # Fixed — required by VGG16
validation_split = 0.2             # Change ratio if needed (e.g., 0.3 for 70/30)
```

### 🔓 Unfreeze VGG16 for Fine-Tuning *(Advanced)*
After initial training, unfreeze the top VGG16 layers for domain-specific fine-tuning:

```python
# Unfreeze last block of VGG16 for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Re-compile with a lower learning rate to avoid catastrophic forgetting
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### ➕ Add Dropout to Reduce Overfitting
```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),          # ← Add this
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),          # ← And this
    tf.keras.layers.Dense(5, activation='softmax')
])
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

```
1. 🍴 Fork the repository
2. 🌿 Create your branch     →  git checkout -b feature/amazing-feature
3. 💾 Commit your changes    →  git commit -m "Add amazing feature"
4. 📤 Push to your branch    →  git push origin feature/amazing-feature
5. 🔃 Open a Pull Request
```

Please make sure your code follows PEP8 standards and includes appropriate comments.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Free to use, modify, and distribute with attribution.
```

---

## 🙏 Acknowledgements

| Resource | Description |
|----------|-------------|
| [📄 VGG16 Paper](https://arxiv.org/abs/1409.1556) | Simonyan & Zisserman, *"Very Deep Convolutional Networks"*, 2014 |
| [🔧 TensorFlow](https://www.tensorflow.org/) | End-to-end machine learning platform |
| [🌐 ImageNet](https://www.image-net.org/) | Large-scale visual recognition dataset for pretraining |
| [📊 scikit-learn](https://scikit-learn.org/) | Machine learning metrics and utilities |

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2E7D32,100:4CAF50&height=120&section=footer" width="100%"/>

**⭐ If this project helped you, please consider giving it a star!**

<br/>

*Made with ❤️ for smarter, sustainable agriculture*

</div>
