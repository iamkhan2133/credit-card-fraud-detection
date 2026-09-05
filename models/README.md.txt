# Saved Models

This directory is used to store the trained machine learning model selected for fraud detection.

The model can be saved using Joblib.

Example:

```python
import joblib

joblib.dump(best_model, "best_fraud_detection_model.pkl")