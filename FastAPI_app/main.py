from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved pipeline
pipeline = joblib.load('prediction_pipeline.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define the request body
class PredictionRequest(BaseModel):
    features: list

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Alzheimer's Detection API"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert the input features to a NumPy array
        features = np.array(request.features)

        # Validate feature length for each input
        expected_features = 32  # Number of input features expected by the model
        if features.shape[1] != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature length: Expected {expected_features} features per input, but received {features.shape[1]}"
            )

        # Predict for all inputs
        predictions = pipeline.predict(features)
        probabilities = pipeline.predict_proba(features)[:, 1]

        # Create a list of results, including the custom message
        results = []
        for pred, prob in zip(predictions, probabilities):
            message = (
                "The patient is suffering with Alzheimer's."
                if pred == 1
                else "The patient is not suffering with Alzheimer's."
            )
            results.append({
                "prediction": int(pred),
                "probability": float(prob),
                "message": message
            })

        # Return the results for all inputs
        return {"results": results}

    except Exception as e:
        # Catch any other unexpected exceptions and return a 500 error
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
