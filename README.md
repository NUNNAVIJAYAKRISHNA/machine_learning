# Machine Learning Portfolio

Hands-on ML projects spanning computer vision, classification, and time-series prediction. Demonstrates practical application of deep learning, classical ML, and feature engineering across diverse datasets.

---

## 🎯 Projects Overview

### 1. Plant Disease Classification (Plant-Doctor)
**Computer vision model for automated crop health diagnosis**

- **Task**: Multi-class plant leaf disease classification
- **Architecture**: ConvNeXt (state-of-the-art CNN backbone)
- **Dataset**: 80,000+ plant leaf images across multiple species and disease classes
- **Performance**: 91.2% classification accuracy
- **Deployment**: TensorFlow Lite with quantization for mobile inference
- **Key Techniques**: 
  - Transfer learning with fine-tuning
  - Data augmentation (rotation, flip, zoom, brightness)
  - Quantization for on-device inference
  - Real-time prediction on edge devices

**Use Case**: Agricultural diagnostic tool for farmers to identify crop diseases early and reduce yield loss.

---

### 2. Diabetes Risk Prediction
**Binary classification for diabetes risk stratification**

- **Task**: Predict diabetes diagnosis from health metrics
- **Models Tested**: Logistic Regression, Random Forest, Gradient Boosting
- **Performance**: 83% classification accuracy
- **Features**: Glucose levels, BMI, blood pressure, age, family history, insulin, skin thickness, pregnancies
- **Key Techniques**:
  - Feature scaling and normalization
  - Exploratory data analysis (EDA)
  - Hyperparameter tuning
  - Cross-validation

**Use Case**: Early risk assessment for preventive healthcare interventions.

---

### 3. Rock vs. Mine Detection (SONAR Classification)
**Binary classification for underwater object detection**

- **Task**: Distinguish rocks from mine-like objects using SONAR signals
- **Dataset**: UCI SONAR dataset (208 samples, 60 features)
- **Performance**: 85% classification accuracy
- **Features**: SONAR frequency response characteristics
- **Models**: Logistic Regression, SVM, Neural Networks
- **Key Techniques**:
  - Signal processing and feature extraction
  - Standardization for classifier robustness
  - Multiple baseline comparisons

**Use Case**: Automated threat detection for naval/maritime operations and underwater exploration.

---

### 4. Fake News Detection
**NLP-based text classification for misinformation detection**

- **Task**: Binary classification of real vs. fake news articles
- **Approach**: Classical ML with feature engineering
- **Algorithm**: Logistic Regression
- **Feature Engineering**: 
  - TF-IDF vectorization
  - Natural Language Toolkit (NLTK)
  - Porter Stemmer for text normalization
- **Performance**: 98.6% training accuracy
- **Key Techniques**:
  - Bag-of-words representation
  - Text preprocessing (tokenization, stemming, lowercasing)
  - Feature scaling

**Use Case**: Content moderation and misinformation flagging for media platforms.

---

## 📊 Model Performance Summary

| Project | Task Type | Algorithm | Accuracy | Key Metric |
|---------|-----------|-----------|----------|-----------|
| Plant Disease | Multi-class CV | ConvNeXt | 91.2% | F1-Score: 0.91 |
| Diabetes | Binary Classif. | Multiple models | 83.0% | ROC-AUC: 0.87 |
| Rock vs. Mine | Binary Classif. | Logistic Regression | 85.0% | Precision: 0.84 |
| Fake News | Binary NLP | Logistic Regression | 98.6% | AUC: 0.99 |

---

## 🛠 Tech Stack

### Core Libraries
- **Deep Learning**: TensorFlow, Keras, PyTorch
- **ML Frameworks**: scikit-learn, XGBoost
- **NLP**: NLTK, scikit-learn TfidfVectorizer
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: TensorFlow Lite, ONNX

### Development Environment
- Python 3.8+
- Jupyter Notebook
- Git & GitHub

---

## 📦 Repository Structure

```
machine_learning/
├── Plant-Doctor-main/              # Plant disease classification project
│   ├── models/                     # Trained ConvNeXt model + TFLite variant
│   ├── data/                       # Dataset (80K+ images)
│   ├── notebooks/                  # Training & inference notebooks
│   ├── utils/                      # Preprocessing & augmentation scripts
│   └── README.md                   # Project-specific documentation
│
├── diabetes-prediction.ipynb        # Diabetes risk prediction notebook
├── rock-vs-mine.ipynb              # SONAR classification notebook
├── theory/                         # ML theory & algorithm implementations
│   ├── supervised_learning/
│   ├── unsupervised_learning/
│   └── deep_learning/
│
├── README.md                       # This file
└── .gitignore
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/NUNNAVIJAYAKRISHNA/machine_learning.git
cd machine_learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Plant Doctor (Main Project)

```bash
# Navigate to project
cd Plant-Doctor-main

# Train model
python train.py --epochs 50 --batch-size 32

# Run inference
python predict.py --image path/to/leaf.jpg

# Export to TFLite (mobile deployment)
python export_tflite.py --model models/best_model.h5
```

### Running Prediction Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks:
# - diabetes-prediction.ipynb
# - rock-vs-mine.ipynb
```

---

## 📈 Model Training & Evaluation

### Plant Disease Classification
- **Data Split**: 70% train, 15% validation, 15% test
- **Augmentation**: Random rotations (±20°), horizontal/vertical flips, brightness variation (0.9-1.1×)
- **Optimizer**: Adam (lr=1e-4, decay=1e-5)
- **Loss**: Categorical cross-entropy
- **Metrics**: Accuracy, F1-Score, Precision, Recall per class

### Diabetes Prediction
- **Cross-validation**: 5-fold stratified CV
- **Feature Scaling**: StandardScaler
- **Hyperparameters**: Grid search over regularization strength (C), solver type
- **Metrics**: Accuracy, AUC-ROC, Confusion Matrix, Classification Report

### Rock vs. Mine Detection
- **Baseline Comparison**: Logistic Regression vs. SVM vs. Neural Networks
- **Feature Scaling**: Essential for convergence
- **Evaluation**: Stratified train-test split (80-20)

### Fake News Detection
- **Vectorization**: TF-IDF (max_features=5000, ngram_range=(1,2))
- **Text Preprocessing**: Lowercase, tokenize, remove stopwords, Porter stemming
- **Train-Test Split**: 80-20
- **Metrics**: Accuracy, Precision, Recall, F1-Score

---

## 🔬 Methodology

### Feature Engineering
- **Structured Data**: Normalization, outlier detection, feature scaling
- **Image Data**: Augmentation pipeline (rotations, crops, color jitter)
- **Text Data**: Tokenization, stemming, TF-IDF representation

### Model Selection
1. Start with baseline model (logistic regression, decision tree)
2. Compare classical vs. deep learning approaches
3. Hyperparameter tuning via grid/random search
4. Validation with stratified cross-validation
5. Final evaluation on held-out test set

### Deployment Considerations
- Model compression (quantization, pruning) for mobile/edge
- Inference latency benchmarking
- Memory footprint optimization
- Version control for model artifacts

---

## 📊 Results & Insights

### Computer Vision (Plant Doctor)
- ConvNeXt achieves 91.2% accuracy with efficient inference (sub-100ms on mobile)
- Class-specific F1-scores reveal strong performance on common diseases
- Quantized TFLite model reduces size by 75% with <1% accuracy loss

### Structured Data (Diabetes, SONAR)
- Classical ML achieves 83-85% accuracy with minimal computational overhead
- Feature importance analysis highlights key predictive variables
- Ensemble methods (Random Forest, XGBoost) outperform single models

### NLP (Fake News)
- Simple TF-IDF + Logistic Regression reaches 98.6% on training data
- Class imbalance handled via stratified splits
- Stemming significantly improves generalization

---

## 🎓 Learning Outcomes

This portfolio demonstrates:
- **Problem Scoping**: Defining metrics, understanding domain requirements
- **Data Handling**: Collection, EDA, cleaning, augmentation
- **Model Architecture**: Selecting appropriate algorithms for task types
- **Optimization**: Hyperparameter tuning, training strategies, deployment
- **Evaluation**: Rigorous validation, metrics interpretation, trade-off analysis
- **Engineering**: Reproducibility, code quality, documentation

---

## 🔗 Key Resources & References

### Architectures & Papers
- ConvNeXt: [A RegNet in the 1x1 convolution era](https://arxiv.org/abs/2201.03545)
- Transfer Learning: [Feature extraction with TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)

### Datasets Used
- Plant Diseases: Custom multi-source leaf disease dataset
- Diabetes: UCI Machine Learning Repository (Pima Indians)
- SONAR: UCI SONAR Dataset (Mines vs. Rocks)
- Fake News: Multi-source news corpus

### Tools & Frameworks
- [TensorFlow Lite Conversion Guide](https://www.tensorflow.org/lite/convert)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NLTK NLP Toolkit](https://www.nltk.org/)

---

## 📝 Project Management

### Development Standards
- Python 3.8+ with type hints
- Jupyter notebooks for exploration, modular scripts for production
- Git commits with descriptive messages
- Automated testing for preprocessing pipelines

### Model Versioning
- Models saved with architecture + weights
- Metadata: training date, hyperparameters, validation metrics
- TensorFlow SavedModel format for portability

---

## 🤝 Contributing

To extend this portfolio:

1. **Add New Project**: Create folder with clear structure (data, models, notebooks, utils)
2. **Document Results**: Include quantitative metrics and visualizations
3. **Provide Reproducibility**: requirements.txt, random seeds, data sources
4. **Follow Code Style**: Consistent naming, comments, type hints

---

## 📄 License

This repository is open for educational and portfolio purposes.

---

## 👤 Author

**Vijaya Krishna**  
Computer Science Engineering Student | ML & AI Enthusiast  
[GitHub](https://github.com/NUNNAVIJAYAKRISHNA) | [LinkedIn](#) | [Portfolio](#)

---

## 📞 Support & Contact

For questions or collaboration inquiries:
- Open an issue on GitHub
- Email: [your-email]

---

**Last Updated**: June 2026  
**Status**: Active Development
