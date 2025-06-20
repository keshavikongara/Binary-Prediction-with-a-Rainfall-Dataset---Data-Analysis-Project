# Binary-Prediction-with-a-Rainfall-Dataset---Data-Analysis-Project

# Binary Prediction with a Rainfall Dataset

## Overview
This project is built for the Kaggle competition **"Playground Series S5E3"**, which focuses on binary classification using a rainfall dataset. The objective is to predict whether rainfall will occur (Yes or No) based on several meteorological features.

The solution uses a **neural network** implemented with **TensorFlow**, alongside other Python libraries like Pandas, NumPy, and Scikit-learn for data preprocessing and evaluation.

---

## ⚙️ What the Code Does

### 1. Setting Up the Environment
- **Data Access**: Uses `kagglehub` to download the competition dataset `playground-series-s5e3`, which includes `train.csv` and `test.csv`.
- **Key Libraries**:
  - `pandas`, `numpy` – data manipulation and calculations
  - `sklearn` – train-test split, feature scaling
  - `tensorflow` – for neural network modeling
- **Reproducibility**: Sets a fixed random seed (`42`) to ensure consistent results across runs.

---

### 2. Data Loading and Preparation
- **Read Data**:
  - `train.csv`: contains features and target labels (`rainfall`)
  - `test.csv`: contains features without labels
- **Prepare Inputs**:
  - Drop `id` and `rainfall` from training data to get features (`X`)
  - Keep `rainfall` as target (`y`)
  - Drop only `id` from test data (`X_test`) and store it for submission
- **Handle Missing Data**: Fills missing values with column-wise means.
- **Feature Scaling**: Applies `StandardScaler` to normalize the features, which helps neural networks perform better.

---

### 3. Train-Validation Split
- Splits training data into 80% training and 20% validation.
- Uses `stratify=y` to maintain the balance between the rainfall and non-rainfall cases.

---

### 4. Model Architecture – Neural Network
Constructed using TensorFlow’s `Sequential` API:

| Layer Type         | Details                                  |
|--------------------|-------------------------------------------|
| Input              | 128 neurons, `ReLU` activation            |
| Batch Normalization| Normalize activations                     |
| Dropout            | 30% to prevent overfitting                |
| Hidden Layer       | 64 neurons, `ReLU` + BatchNorm + Dropout  |
| Hidden Layer       | 32 neurons, `ReLU` + BatchNorm + Dropout  |
| Output             | 1 neuron, `Sigmoid` activation (binary)   |

This structure enables the model to learn complex patterns and generalize well.

---

### 5. Model Compilation & Training
- **Compile Settings**:
  - Optimizer: `Adam`
  - Loss: `BinaryCrossentropy`
  - Metric: `Accuracy`
- **EarlyStopping**: Stops training if validation loss doesn’t improve for 10 consecutive epochs and restores the best weights.
- **Training Setup**:
  - Max epochs: 100
  - Batch size: 32
  - Validation used to monitor overfitting

---

### 6. Evaluation
- Predicts probabilities on validation data.
- Evaluates performance using **ROC AUC Score**, a robust metric for binary classification (closer to 1.0 = better).

---

### 7. Inference and Submission
- Predicts rainfall probabilities on the test set.
- Generates `submission.csv` with:
  - `id` (from test set)
  - `rainfall` (predicted probabilities)
