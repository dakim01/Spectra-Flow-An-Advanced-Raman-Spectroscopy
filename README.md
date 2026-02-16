# Resolving Silicate Mineral Isomorphism using 1D CNNs and NMF


A progressive machine learning approach to quantifying mineral compositions in complex mixtures using Raman spectroscopy. This project tackles the challenging problem of **mineral isomorphism** – distinguishing between minerals with similar crystal structures that produce overlapping spectral signatures.

## 🎯 Project Highlights

- **Complete Pipeline**: 5-phase progressive analysis from data exploration to deep learning
- **Excellent Results**: Achieved R² = 0.9834 average across three minerals
- **Scientific Rigor**: Systematic evaluation of unsupervised, traditional ML, and deep learning approaches
- **Production Ready**: Fully documented code with trained models

## 📊 Results Summary

| Mineral | R² Score | RMSE | Performance |
|---------|----------|------|-------------|
| **Albite** | 0.9897 | 0.0292 | ★★★ Excellent |
| **Quartz** | 0.9895 | 0.0243 | ★★★ Excellent |
| **Orthoclase** | 0.9708 | 0.0382 | ★★★ Excellent |
| **Average** | **0.9834** | **0.0306** | **Excellent** |

## 🔬 The Challenge

**Mineral Isomorphism** refers to minerals with similar crystal structures that produce overlapping spectral signatures in Raman spectroscopy. This project focuses on three feldspars:

- **Albite** (NaAlSi₃O₈) - Sodium feldspar
- **Quartz** (SiO₂) - Silicon dioxide  
- **Orthoclase** (KAlSi₃O₈) - Potassium feldspar

Traditional spectroscopic analysis struggles to distinguish and quantify these minerals in mixtures due to their spectral overlap.

## 🚀 The Progressive Approach

Rather than jumping directly to complex models, this project systematically evaluates different approaches:

### Phase 1: Data Loading & Exploration
- **Dataset**: 303 samples, 1,504 spectral features each
- **Split**: 80/20 train-test split (242/61 samples)
- **Purpose**: Understand data quality, distribution, and characteristics

### Phase 2: Non-Negative Matrix Factorization (NMF)
- **Approach**: Unsupervised decomposition to extract pure mineral spectra
- **Result**: Complete failure (Average R² = -0.117)
- **Lesson**: Unsupervised methods cannot handle spectral overlap

### Phase 3: Random Forest Regression
- **Approach**: Supervised learning with ensemble methods
- **Result**: Strong performance (Average R² = 0.9510)
- **Lesson**: Non-linear supervised learning works well

### Phase 4: NMF + Random Forest Hybrid
- **Approach**: NMF for feature extraction, RF for prediction
- **Result**: Marginal improvement (Average R² = 0.9554)
- **Lesson**: Handcrafted feature engineering has limits

### Phase 5: 1D Convolutional Neural Network
- **Approach**: End-to-end deep learning with multi-scale architecture
- **Result**: Excellent performance (Average R² = 0.9834)
- **Lesson**: Automatic feature learning achieves breakthrough results

## 🏗️ CNN Architecture

The final 1D CNN employs a sophisticated multi-scale design:

```
Input (1504 features)
    ↓
Conv1D (64 filters, kernel=7) → BatchNorm → ReLU → MaxPool
    ↓
Conv1D (128 filters, kernel=5) → BatchNorm → ReLU → MaxPool
    ↓
Conv1D (64 filters, kernel=3) → BatchNorm → ReLU → MaxPool
    ↓
Global Average Pooling
    ↓
Dropout (50%)
    ↓
Dense (3 outputs) → Linear activation
```

**Key Features:**
- **Multi-scale processing**: Decreasing kernel sizes (7→5→3) capture broad patterns to fine details
- **Regularization**: Batch normalization and 50% dropout prevent overfitting
- **End-to-end learning**: Optimizes features specifically for mineral quantification

## 📁 Project Structure

```
raman-spectroscopy-mineral-analysis/
├── data/
│   ├── raw/                    # Original spectral data
│   ├── processed/              # Preprocessed datasets
│   └── README.md               # Data documentation
├── notebooks/
│   └── Raman_Spectroscopy_Analysis.ipynb  # Main analysis notebook
├── src/
│   ├── preprocessing.py        # Data loading and preprocessing
│   ├── models/
│   │   ├── nmf_model.py       # NMF implementation
│   │   ├── rf_model.py        # Random Forest models
│   │   └── cnn_model.py       # 1D CNN architecture
│   ├── evaluation.py          # Model evaluation utilities
│   └── visualization.py       # Plotting functions
├── models/
│   ├── nmf_components.pkl     # Trained NMF model
│   ├── rf_albite.pkl          # Random Forest for Albite
│   ├── rf_quartz.pkl          # Random Forest for Quartz
│   ├── rf_orthoclase.pkl      # Random Forest for Orthoclase
│   └── cnn_model.h5           # Trained CNN model
├── results/
│   ├── figures/               # Generated plots and visualizations
│   └── metrics/               # Performance metrics
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
├── LICENSE
└── README.md
```

## 🛠️ Installation

### Using pip


## 📦 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## 🎓 Usage

### Quick Start with Jupyter Notebook

```bash
jupyter notebook notebooks/Raman_Spectroscopy_Analysis.ipynb
```

The notebook contains all five phases with detailed explanations and visualizations.

### Using Individual Models

#### Random Forest Prediction

```python
from src.models.rf_model import load_rf_models, predict_composition

# Load trained models
models = load_rf_models('models/')

# Predict mineral composition
spectrum = load_spectrum('path/to/spectrum.csv')
composition = predict_composition(models, spectrum)

print(f"Albite: {composition['albite']:.2f}%")
print(f"Quartz: {composition['quartz']:.2f}%")
print(f"Orthoclase: {composition['orthoclase']:.2f}%")
```

#### CNN Prediction

```python
from src.models.cnn_model import load_cnn_model, predict_composition

# Load trained CNN
model = load_cnn_model('models/cnn_model.h5')

# Predict mineral composition
spectrum = load_spectrum('path/to/spectrum.csv')
composition = predict_composition(model, spectrum)

print(f"Albite: {composition['albite']:.2f}%")
print(f"Quartz: {composition['quartz']:.2f}%")
print(f"Orthoclase: {composition['orthoclase']:.2f}%")
```

## 📈 Performance Comparison

| Phase | Method | Avg R² | Avg RMSE | Status |
|-------|--------|--------|----------|--------|
| 2 | NMF Only | -0.117 | 0.2156 | ❌ Failed |
| 3 | Random Forest | 0.9510 | 0.0428 | ✅ Strong |
| 4 | NMF + RF | 0.9554 | 0.0414 | 📈 Marginal |
| 5 | **1D CNN** | **0.9834** | **0.0306** | ⭐ Excellent |

### Mineral-Specific Performance (CNN)

**Albite**: Improved from R² = 0.9626 (RF) to 0.9897 (CNN) - **+2.71%**  
**Quartz**: Improved from R² = 0.9691 (RF) to 0.9895 (CNN) - **+2.04%**  
**Orthoclase**: Improved from R² = 0.9214 (RF) to 0.9708 (CNN) - **+4.94%** ⭐

*Note: Orthoclase showed the largest improvement, demonstrating CNN's ability to extract subtle spectral differences.*

## 🔑 Key Insights

### 1. Progressive Complexity is Essential
Starting with simple methods before advancing to complex ones wasn't wasted effort. Each phase provided crucial insights:
- NMF failure revealed the need for supervision
- Random Forest success proved the problem was solvable
- NMF+RF showed limits of handcrafted features
- CNN demonstrated power of end-to-end learning

### 2. Supervision is Required
Unsupervised methods completely failed due to spectral overlap. The minerals are too similar for blind source separation.

### 3. Task-Specific Feature Learning Wins
Generic dimensionality reduction (NMF) couldn't compete with task-specific feature learning (CNN). When features are learned end-to-end, the model discovers representations optimized for distinguishing isomorphic minerals.

### 4. Domain Knowledge Guides Architecture
The CNN's multi-scale architecture reflects understanding of spectroscopic analysis. Decreasing kernel sizes (7→5→3) mirror how spectroscopists examine spectra hierarchically.

## 🌟 Applications

This technology enables:

- **⚡ Rapid Mineral Analysis**: Automated quantification without manual spectral interpretation
- **✅ Quality Control**: Real-time monitoring of mineral composition in industrial processes
- **🌍 Geological Studies**: High-throughput analysis of rock samples for petrology research
- **⛏️ Resource Exploration**: Characterizing mineral deposits for mining applications
- **🏭 Materials Science**: Process optimization in ceramics and glass manufacturing

## 🔮 Future Directions

- [ ] Expand to additional mineral species (pyroxenes, amphiboles)
- [ ] Multi-technique integration (Raman + IR + XRD)
- [ ] Uncertainty quantification with Bayesian deep learning
- [ ] Real-time inference optimization for field deployment
- [ ] Transfer learning for new mineral groups
- [ ] Interactive web application for mineral analysis

## 🙏 Acknowledgments

- Dataset source and methodology inspiration
- TensorFlow and scikit-learn communities
- Research community working on spectroscopic analysis
- Contributors and collaborators


<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the scientific community

</div>
