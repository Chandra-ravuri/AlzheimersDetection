# AlzheimersDetection
## Model Comparison and Selection

We evaluated two machine learning models for Alzheimer's detection:

1. **Logistic Regression**:
   - A simpler linear model.
   - Achieved moderate performance with:
     - Precision: **0.62**
     - Recall: **0.67**
     - ROC-AUC: **0.89**

2. **XGBoost**:
   - A powerful ensemble learning model.
   - Achieved significantly better results with:
     - Precision: **0.96**
     - Recall: **0.92**
     - ROC-AUC: **0.95**

### Why XGBoost?
- **Higher Recall:** XGBoost identified more Alzheimer’s patients, reducing false negatives.
- **Better Precision:** It reduced the number of false positives compared to Logistic Regression.
- **Robustness:** XGBoost performed better with the dataset's complexity and feature interactions.

Based on these results, we chose XGBoost as the final model for deployment in the FastAPI application.

## FastAPI Application

The final XGBoost model was deployed using FastAPI. The API allows:
- Predictions for individual or multiple patients.
- JSON responses with probabilities and custom messages.

### Running the Application
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
