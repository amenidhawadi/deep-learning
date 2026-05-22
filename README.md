# deep-learning
# Heart Disease Prediction using Deep Learning

## Problem Statement
Predict the presence of heart disease (binary classification) from patient medical records. Early detection can significantly reduce mortality.

## Dataset
- **Source**: Provided `heart.csv` (918 records, 11 features + target).
- **Features**:
  - `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`
- **Target**: `HeartDisease` (0 = No, 1 = Yes)
- **Missing/Invalid**: RestingBP and Cholesterol contain zeros → replaced with median.

## Approach
- **Preprocessing**:
  - Label encoding for categorical variables.
  - Standard scaling for numerical features.
  - SMOTE oversampling to handle class imbalance.
- **Models compared**:
  - Multi‑Layer Perceptron (MLP)
  - 1‑Dimensional Convolutional Neural Network (1D-CNN)
  - Long Short‑Term Memory (LSTM)
  - Transformer for tabular data
- **Training**:
  - Loss: Binary Crossentropy
  - Optimizer: Adam with learning rate scheduling
  - Early stopping (patience = 15)
  - Batch size = 32, max epochs = 100

## Results (on test set)
| Model        | Accuracy | AUC    |
|--------------|----------|--------|
| MLP          | 0.84     | 0.89   |
| 1D-CNN       | 0.87     | 0.91   |
| LSTM         | 0.86     | 0.90   |
| Transformer  | **0.89** | **0.93** |

- The **Transformer** model achieved the highest AUC (0.93), indicating excellent discrimination.
- Feature importance (via SHAP) highlighted `Oldpeak`, `MaxHR`, and `ChestPainType` as top predictors.

## How to Run the Code
1. **Clone the repository** (or copy the notebook).
2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow shap
