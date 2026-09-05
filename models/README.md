# Saved Models

This directory contains the trained machine learning model selected as the best-performing model for credit card fraud detection.

## Best Model

The best model is selected based on the evaluation results, with particular consideration given to F1-score, ROC-AUC, precision, and recall.

## Model Format

The trained model is saved using Joblib:

`best_fraud_detection_model.pkl`

## Load the Model

```python
import joblib

model = joblib.load("models/best_fraud_detection_model.pkl")
```

## Save the Model

```python
import joblib

joblib.dump(best_model, "models/best_fraud_detection_model.pkl")
```

The saved model can be reused later for fraud prediction without retraining the model.
