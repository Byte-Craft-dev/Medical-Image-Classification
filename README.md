# Medical Image Classification: Comparative ML Approach

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Overview

A comprehensive comparison of four machine learning approaches for 11-class medical image classification:
- **Support Vector Machine (SVM)** - 94.9% accuracy
- **K-Nearest Neighbors (KNN)** - 93.0% accuracy
- **Convolutional Neural Network (CNN)** - 85.0% accuracy
- **Generative Adversarial Network (GAN)** - 50.6% accuracy

## 🎯 Key Achievements

- ✅ **Best Performance:** 94.9% accuracy with SVM (0.998 AUC)
- ✅ **Feature Engineering:** 357-dimensional feature space (HOG + LBP + Statistics)
- ✅ **Class Imbalance Handling:** GAN-based synthetic data generation (4.7:1 → 1.5:1)
- ✅ **Comprehensive Evaluation:** ROC curves, confusion matrices, per-class metrics
- ✅ **Optimized Hyperparameters:** Grid Search with 3-fold cross-validation

## 📊 Performance Comparison

| Model | Accuracy | AUC | Training Time | Key Feature |
|-------|----------|-----|---------------|-------------|
| **SVM** | 94.9% | 0.998 | 12 min | Near-perfect classification |
| **KNN** | 93.0% | 0.977 | Instant | No training required |
| **CNN** | 85.0% | 0.987 | 5 min | Automatic feature learning |
| **GAN** | 50.6% | 0.892 | 10 min | Synthetic data generation |

## 🛠️ Tech Stack

**Languages & Frameworks:**
- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- OpenCV
- scikit-image

**Libraries:**
- NumPy, Pandas - Data manipulation
- Matplotlib, Seaborn - Visualization
- tqdm - Progress bars
- joblib - Model persistence

## 📁 Project Structure

```
medical-image-classification/
├── notebooks/              # Jupyter notebooks for each approach
│   ├── 01_KNN_Classification.ipynb
│   ├── 02_SVM_Classification.ipynb
│   ├── 03_CNN_Classification.ipynb
│   └── 04_GAN_Classification.ipynb
├── results/               # Visualizations and metrics
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── performance_plots/
├── models/                # Saved trained models
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   └── cnn_model.h5
├── src/                   # Source code modules
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── evaluation.py
├── docs/                  # Documentation and presentations
├── requirements.txt       # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8 or higher
pip or conda package manager
Jupyter Notebook
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Securedbytes/medical-image-classification.git
cd medical-image-classification
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
   - Contact the author for dataset access (see contact section)
   - Place dataset files in the appropriate directory structure

### Usage

**Run the Jupyter notebooks:**

```bash
jupyter notebook
```

Then open and run the notebooks in order:
1. `01_KNN_Classification.ipynb` - K-Nearest Neighbors approach
2. `02_SVM_Classification.ipynb` - Support Vector Machine approach
3. `03_CNN_Classification.ipynb` - Convolutional Neural Network approach
4. `04_GAN_Classification.ipynb` - Generative Adversarial Network approach

## 📈 Results

### SVM Results (Best Performance)

**Key Metrics:**
- **Accuracy:** 94.9%
- **Precision:** 0.949
- **Recall:** 0.949
- **F1-Score:** 0.949
- **AUC:** 0.998

**Confusion Matrix Highlights:**
- Excellent performance across all 11 classes
- Minimal false positives/negatives
- Robust to class imbalance

### Feature Engineering

**357-Dimensional Feature Vector:**

1. **HOG (81 features):** Histogram of Oriented Gradients
   - 9 orientations, 8×8 pixel cells
   - Captures shape and edge information

2. **LBP (235 features):** Local Binary Patterns
   - Radius 2, 16 points
   - Captures texture information

3. **Statistical (41 features):**
   - Mean, standard deviation, median, quartiles
   - Skewness, kurtosis, 32-bin histogram
   - Captures intensity distribution

## 🔬 Methodology

### 1. Data Preprocessing
- **Image resizing:** 32×32 pixels
- **Normalization:** 0-1 range
- **Class distribution analysis**
- **Train/test split:** 80/20

### 2. Feature Extraction (SVM/KNN)
- HOG for shape features
- LBP for texture features
- Statistical features for intensity distribution
- Feature concatenation and normalization

### 3. Model Training

**SVM:**
- Kernel: Radial Basis Function (RBF)
- Hyperparameter tuning: Grid Search
- Cross-validation: 3-fold

**KNN:**
- K=5 neighbors
- Distance metric: Euclidean
- Weighting: Distance-weighted

**CNN:**
- Architecture: 3 convolutional blocks
- Dropout: 0.5 for regularization
- Optimizer: Adam
- Epochs: 50 with early stopping

**GAN:**
- Generator: 4-layer deep network
- Discriminator: 4-layer classifier
- Training: 20 epochs adversarial learning
- Application: Synthetic data generation for minority classes

### 4. Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Visualizations:** ROC curves and AUC
- **Analysis:** Confusion matrices, per-class performance
- **Comparison:** Cross-model performance benchmarking

## 💡 Key Insights

1. **Feature Engineering Matters:** Hand-crafted features (SVM/KNN) outperformed automatic learning (CNN) in this low-resolution scenario
2. **Class Imbalance Challenge:** GAN successfully generated synthetic data, reducing imbalance from 4.7:1 to 1.5:1
3. **SVM's Kernel Trick:** Non-linear separation crucial for achieving 94.9% accuracy
4. **Resolution Limitations:** 32×32 pixel resolution is a bottleneck for deep learning approaches
5. **Model Interpretability:** Traditional ML models offer better explainability for healthcare applications

## 🎓 Lessons Learned

- Traditional ML with good feature engineering can outperform deep learning on small datasets
- Class imbalance requires multiple strategies (class weights, synthetic data generation)
- Low-resolution images (32×32) limit the effectiveness of deep learning
- Model interpretability is crucial for healthcare applications
- Computational efficiency varies significantly across approaches

## 🔮 Future Work

- [ ] Experiment with higher resolution images (128×128, 224×224)
- [ ] Implement ensemble methods combining multiple models
- [ ] Apply transfer learning with pre-trained networks (ResNet, VGG)
- [ ] Explore attention mechanisms for interpretability
- [ ] Deploy model as a web application
- [ ] Conduct clinical validation studies

## 📚 Dataset

The dataset consists of 11 classes of medical images, resized to 32×32 pixels for computational efficiency. For dataset access, please contact the author.

**Dataset Statistics:**
- Total images: [60,000]
- Classes: 11
- Resolution: 32×32 pixels
- Split: 80% training, 20% testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Tharushi Karawgoda**
- GitHub: [@Byte-Craft-dev](https://github.com/Byte-Craft-dev)
- LinkedIn: [Tharushi Karawgoda](http://www.linkedin.com/in/tharushi-navodya-)
- Email: tharushi123navo@gmail.com

## 🙏 Acknowledgments
Project Developed  by @Securedbytes
- Dataset providers and medical imaging community
- TensorFlow, scikit-learn, and OpenCV development teams
- Research papers and tutorials that inspired this work

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
  author = {Karawgoda, Tharushi},
  title = {Medical Image Classification: Comparative ML Approach},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Byte-Craft-dev/Medical-Image-Classification}
}
```

---

⭐ **If you find this project useful, please consider giving it a star!**

## 📞 Questions or Feedback?

Feel free to open an issue or reach out via email. Contributions and suggestions are welcome!
