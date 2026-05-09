# Maternal Risk Prediction using Machine Learning

## Project Overview

This project focuses on predicting maternal health risk levels using machine learning algorithms based on medical indicators collected during pregnancy.

The main objective is to classify patients into different maternal risk categories using supervised machine learning models and compare their performances through comprehensive evaluation metrics.

The project includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing pipeline
- Multiple machine learning classification models
- Model evaluation and comparison
- Feature importance analysis
- Professional project modularization using Python scripts

---

# Dataset Information

The dataset contains maternal health-related measurements such as:

- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Blood Sugar (BS)
- Body Temperature
- BMI
- Previous Complications
- Preexisting Diabetes
- Gestational Diabetes
- Mental Health
- Heart Rate

Target Variable:

- Risk Level
  - Low Risk
  - High Risk

Dataset Size:

- 1205 rows
- 12 columns

---

# Project Structure

```bash
MATERNAL-RISK-PREDICTION/
│
├── data/
│   └── Dataset.csv
│
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
│
├── presentation/
│
├── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluation.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Technologies Used

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

# Exploratory Data Analysis (EDA)

The EDA phase included:

- Missing value analysis
- Distribution analysis
- Boxplot analysis
- Correlation analysis
- Outlier detection
- QQ plot analysis
- Risk-level-based comparisons

Important findings:

- Blood Sugar (BS) showed strong relationship with maternal risk level
- Preexisting Diabetes had significant predictive influence
- BMI and Heart Rate contributed strongly to classification
- Several variables contained outliers but remained medically meaningful

---

# Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Median imputation for numerical features
- Mode imputation for target variable
- Target variable encoding
- Train-test splitting
- Feature scaling using StandardScaler

Target Encoding:

| Risk Level | Encoded Value |
| ---------- | ------------- |
| Low        | 0             |
| High       | 1             |

---

# Machine Learning Models

The following classification models were implemented:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest

---

# Model Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- ROC-AUC Score
- Cross Validation Mean
- Confusion Matrix
- Classification Report
- Feature Importance Analysis

---

# Final Model Performance

| Model               | Accuracy | ROC-AUC | CV Mean |
| ------------------- | -------- | ------- | ------- |
| Random Forest       | 0.9834   | 0.9863  | 0.9823  |
| KNN                 | 0.9668   | 0.9671  | 0.9678  |
| Decision Tree       | 0.9627   | 0.9581  | 0.9595  |
| SVM                 | 0.9585   | 0.9602  | 0.9730  |
| Logistic Regression | 0.9544   | 0.9531  | 0.9730  |

---

# Best Performing Model

## Random Forest

Random Forest achieved the best overall performance with:

- Accuracy: 98.34%
- ROC-AUC: 98.63%
- Cross Validation Mean: 98.23%

The model demonstrated:

- Strong generalization capability
- High classification reliability
- Excellent discrimination performance
- Stable cross-validation results

---

# Feature Importance Analysis

The most influential features identified by Random Forest were:

1. Preexisting Diabetes
2. Blood Sugar (BS)
3. BMI
4. Heart Rate
5. Mental Health

These findings indicate that diabetes-related indicators and metabolic measurements play major roles in maternal risk prediction.

---

# Visualizations

## EDA Visualizations

The project includes:

- Distribution plots
- Boxplots
- Correlation heatmap
- Risk-group comparisons
- QQ plots

## Model Training Visualizations

The project also includes:

- Confusion matrices
- Feature importance graph
- Model accuracy comparison chart

---

# Example Visualizations

## Model Accuracy Comparison

```markdown
Add image here:
MODEL TRAINING/7 - Model Accuracy Comparison.png
```

## Random Forest Feature Importance

```markdown
Add image here:
MODEL TRAINING/6 - Random Forest Feature Importance.png
```

## Correlation Heatmap

```markdown
Add image here:
EDA/11 - Correlation Heatmap.png
```

---

# How to Run

## Clone Repository

```bash
git clone <repository-link>
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Run Preprocessing Pipeline

```bash
python src/preprocessing.py
```

## Train Models

```bash
python src/train_models.py
```

## Run Evaluation Pipeline

```bash
python src/evaluation.py
```

---

# Future Improvements

Possible future developments:

- Hyperparameter tuning
- Deep learning approaches
- Web deployment using Flask or FastAPI
- Real-time maternal risk prediction system
- Additional medical feature integration
- Multi-class risk classification

---

# Author

Rümeysa YAVUZKANAT

Emine Sena TOP

Computer Engineering Senior Students

Machine Learning & Software Development Enthusiast
