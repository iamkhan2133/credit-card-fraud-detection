# Credit Card Fraud Detection using Machine Learning

## Devixo Solutions AI/ML Internship – Week 3 Task 03

An end-to-end machine learning project for detecting fraudulent credit card transactions using multiple machine learning algorithms.

---

## Project Overview

Credit card fraud detection is an important real-world machine learning problem because fraudulent transactions represent a very small percentage of total transactions.

The objective of this project is to develop a complete machine learning workflow starting from data preparation and feature engineering to model training, evaluation, comparison, and model selection.

The project evaluates multiple classification algorithms and compares their performance using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

---

## Objective

The main objectives of this project are:

- Load and analyze a real-world credit card transaction dataset.
- Handle missing values and duplicate records.
- Perform feature engineering.
- Select relevant features.
- Scale numerical features where required.
- Handle severe class imbalance.
- Train multiple machine learning models.
- Evaluate model performance using appropriate classification metrics.
- Analyze feature importance.
- Compare different machine learning models.
- Save the best-performing model.
- Develop a reusable machine learning workflow.

---

## Dataset

The dataset used in this project is a real-world credit card transaction dataset.

### Dataset Information

- Total records: 1,296,675
- Total columns: 23
- Target variable: `is_fraud`
- Legitimate transaction: `0`
- Fraudulent transaction: `1`

The dataset contains highly imbalanced classes because fraudulent transactions represent a very small proportion of all transactions.

### Class Distribution

- Legitimate transactions: 1,289,169
- Fraudulent transactions: 7,506

Fraudulent transactions represent approximately 0.58% of the complete dataset.

The raw dataset is not included in this repository because of its large size.

---

## Project Workflow

The project follows the following machine learning workflow:

1. Data Collection
2. Data Loading
3. Data Quality Analysis
4. Missing Value Analysis
5. Duplicate Detection
6. Feature Engineering
7. Feature Selection
8. Class Imbalance Handling
9. Data Preprocessing
10. Train-Test Split
11. Model Training
12. Model Evaluation
13. Model Comparison
14. Feature Importance Analysis
15. Model Optimization
16. Best Model Selection
17. Model Saving

---

## Data Preparation

### Missing Values

The dataset was checked for missing values.

Result:

- Missing values: 0

Therefore, no missing-value imputation was required.

### Duplicate Records

The dataset was checked for duplicate rows.

Result:

- Duplicate records: 0

Therefore, no duplicate records were removed.

### Infinite Values

The dataset was also checked for infinite numerical values.

Result:

- Infinite values: 0

---

## Feature Engineering

Several new features were created to improve the model's ability to identify fraudulent transactions.

The following engineered features were created:

- `Amount_Log`
- `transaction_hour`
- `transaction_day_of_week`
- `transaction_month`
- `transaction_day`
- `transaction_week_of_year`
- `transaction_quarter`
- `age`
- `distance_to_merchant`

### Amount Transformation

The transaction amount was transformed using a logarithmic transformation:

`Amount_Log`

This helps reduce the effect of highly skewed transaction amounts.

### Time-Based Features

The transaction timestamp was converted into multiple temporal features including:

- Transaction hour
- Day of week
- Month
- Day
- Week of year
- Quarter

### Age Feature

Customer age was derived from the transaction date and date of birth.

### Distance Feature

The geographical distance between the transaction location and merchant location was calculated.

---

## Feature Selection

The following columns were removed because they were identifiers, redundant information, personally identifiable information, or high-cardinality fields:

- `Unnamed: 0`
- `cc_num`
- `trans_num`
- `trans_date_trans_time`
- `dob`
- `unix_time`
- `first`
- `last`
- `street`
- `city`
- `state`
- `zip`
- `job`

The following categorical features were retained and encoded:

- `merchant`
- `category`
- `gender`

---

## Handling Class Imbalance

The dataset contains a severe class imbalance.

Fraudulent transactions represent only approximately 0.58% of the complete dataset.

To reduce the effect of class imbalance:

- Decision Tree used `class_weight='balanced'`
- Random Forest used `class_weight='balanced'`
- XGBoost used `scale_pos_weight`
- Gradient Boosting was trained using the available standard configuration

For XGBoost, the calculated class weight was approximately:

`scale_pos_weight = 171.75`

Because fraud detection is a highly imbalanced classification problem, Accuracy alone is not considered sufficient for model selection.

---

## Machine Learning Models

The following models were trained and evaluated:

### 1. Decision Tree

A Decision Tree classifier was trained using balanced class weights.

### 2. Random Forest

A Random Forest classifier was trained using balanced class weights.

### 3. Gradient Boosting

A Gradient Boosting classifier was trained to identify nonlinear relationships between transaction features and fraud.

### 4. XGBoost

XGBoost was trained with class imbalance handling using `scale_pos_weight`.

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.997652 | 0.788486 | 0.812125 | 0.800131 | 0.905428 |
| Random Forest | 0.998087 | 0.987391 | 0.678215 | 0.804107 | 0.992117 |
| Gradient Boosting | 0.997370 | 0.825776 | 0.691539 | 0.752719 | 0.970936 |
| XGBoost | 0.991756 | 0.410003 | 0.966689 | 0.575794 | 0.998807 |

---

## Model Analysis

### Decision Tree

The Decision Tree achieved:

- Accuracy: 99.77%
- Precision: 78.85%
- Recall: 81.21%
- F1-Score: 80.01%
- ROC-AUC: 90.54%

It provides a reasonable balance between precision and recall and has a relatively simpler model structure.

---

### Random Forest

The Random Forest achieved:

- Accuracy: 99.81%
- Precision: 98.74%
- Recall: 67.82%
- F1-Score: 80.41%
- ROC-AUC: 99.21%

Random Forest achieved the highest precision and highest F1-score among the evaluated models.

However, its recall is lower than the Decision Tree and XGBoost, meaning it misses a larger number of fraudulent transactions.

---

### Gradient Boosting

The Gradient Boosting model achieved:

- Accuracy: 99.74%
- Precision: 82.58%
- Recall: 69.15%
- F1-Score: 75.27%
- ROC-AUC: 97.09%

The model achieved strong precision and ROC-AUC but lower recall than the Decision Tree and XGBoost.

---

### XGBoost

XGBoost achieved:

- Accuracy: 99.18%
- Precision: 41.00%
- Recall: 96.67%
- F1-Score: 57.58%
- ROC-AUC: 99.88%

XGBoost achieved the highest recall and highest ROC-AUC.

The high recall means that the model successfully identifies most fraudulent transactions.

However, its precision is considerably lower, resulting in more false-positive predictions.

---

## Confusion Matrix Analysis

### Decision Tree

True Negatives: 257,507  
False Positives: 327  
False Negatives: 282  
True Positives: 1,219

### Random Forest

True Negatives: 257,821  
False Positives: 13  
False Negatives: 483  
True Positives: 1,018

### Gradient Boosting

True Negatives: 257,615  
False Positives: 219  
False Negatives: 463  
True Positives: 1,038

### XGBoost

True Negatives: 255,746  
False Positives: 2,088  
False Negatives: 50  
True Positives: 1,451

---

## Why Accuracy Is Not Enough

The dataset is highly imbalanced, with legitimate transactions greatly outnumbering fraudulent transactions.

Therefore, a model could achieve very high accuracy while still failing to detect a significant number of fraudulent transactions.

For fraud detection, Recall is particularly important because false negatives represent fraudulent transactions incorrectly classified as legitimate.

Precision is also important because a high number of false positives can negatively affect legitimate customers.

Therefore, this project considers Precision, Recall, F1-Score, ROC-AUC, and the Confusion Matrix along with Accuracy.

---

## Model Selection

There is no single model that is best for every fraud-detection objective.

### Highest Accuracy

Random Forest:

99.81%

### Highest Precision

Random Forest:

98.74%

### Highest Recall

XGBoost:

96.67%

### Highest F1-Score

Random Forest:

80.41%

### Highest ROC-AUC

XGBoost:

99.88%

For a fraud-detection system where missing fraudulent transactions is considered more costly than generating additional alerts, XGBoost is a strong candidate because of its very high recall and ROC-AUC.

Random Forest is preferable when reducing false positives and maintaining high precision is the primary objective.

---

## Feature Importance

Feature importance analysis was performed to identify the features that contribute most to fraud prediction.

The feature importance results are available in the `results/` directory.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Google Colab
- GitHub

---

## Project Structure

```text
credit-card-fraud-detection-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── results/
│   ├── model_comparison.csv
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_gradient_boosting.png
│   ├── confusion_matrix_xgboost.png
│   └── feature_importance.png
│
├── reports/
│   └── Credit_Card_Fraud_Detection_Report.pdf
│
└── screenshots/
    ├── dataset.png
    ├── preprocessing.png
    ├── class_distribution.png
    ├── model_training.png
    └── model_results.png
