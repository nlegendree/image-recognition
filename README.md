# Image Recognition Pipeline

A comprehensive computer vision project implementing multiple feature extraction methods and classification techniques for image recognition and retrieval tasks.

## 🎯 Overview

This project provides an end-to-end image recognition pipeline featuring:
- **4 Feature Extraction Methods**: Bag of Words (BoW), Spatial Pyramid Matching, VGG13, VGG19
- **3 Classification Models**: Support Vector Machine (SVM), Random Forest, Fully Connected Neural Network
- **Image Retrieval System**: Content-based retrieval using visual feature similarity

## 📊 Dataset

The pipeline is designed for the [Caltech20 dataset](https://data.caltech.edu/records/mzrjq-6wc02), containing 20 object categories.

## 🏗️ Project Structure

```
.
├── classifiers/                 # Classification models
│   ├── fc.py                    # Fully Connected Neural Network
│   ├── random_forest.py         # Random Forest classifier
│   └── svm.py                   # Support Vector Machine
│
├── data/                        # Dataset directory
│   └── caltech20/               # Place Caltech20 dataset here
│
├── features/                    # Feature extraction modules
│   ├── bow.py                   # Bag of Words implementation
│   ├── spatial_pyramid.py       # Spatial Pyramid Matching
│   └── vgg.py                   # VGG-based deep features
│
├── output/                      # Generated results (gitignored)
│   ├── confmats/                # Confusion matrices
│   └── retrieval/               # Image retrieval results
│
├── retrieval/
│   └── retrieval.py             # Image retrieval implementation
│
├── utils/
│   ├── data_loader.py           # Dataset loading utilities
│   └── plot.py                  # Visualization tools
│
└── main.py                      # Main pipeline script
```

## 🚀 Getting Started

### Prerequisites

Python 3.8+ required.

```bash
pip install -r requirements.txt
```

### Installation

1. Clone this repository

2. Download the [Caltech20 dataset](https://data.caltech.edu/records/mzrjq-6wc02) and place it in `data/caltech20/`

3. Run the pipeline
```bash
python main.py
```

## 📈 Features

### Feature Extraction
- **Bag of Words (BoW)**: SIFT descriptors with K-Means clustering
- **Spatial Pyramid Matching**: Multi-scale BoW with spatial information
- **VGG13/VGG19**: Deep convolutional features from pretrained networks

### Classification
All three classifiers are evaluated on each feature extraction method, with confusion matrices and accuracy metrics generated automatically.

### Image Retrieval
Content-based image retrieval using cosine similarity on extracted features, with top-k results visualization.

## 📁 Output

The pipeline generates:
- Feature vectors saved as `.npy` files
- Confusion matrices for each feature/classifier combination
- Image retrieval visualizations showing top-15 similar images
- Classification accuracy reports

## 🛠️ Customization

You can modify hyperparameters in `main.py`:
- Number of visual words (BoW vocabulary size)
- PCA dimensions
- Spatial pyramid levels
- Classifier parameters
