
# main.py

import time
import pickle
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, Extra
import uvicorn
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Setup logger
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Salary Prediction API",  # Change to appropriate Title based on JSON block
    description="FastAPI service for serving predictions and evaluation metrics for the ML model.",
    version="1.0.0"
)

# Allow CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Pydantic Models for Validation
# -------------------------------

class PredictionRequest(BaseModel):
    # Accepts an arbitrary number of inputs in key-value form.
    # Example: { "inputs": { "feature1": 0.5, "feature2": "sample text", "feature3": 10 } }
    inputs: Dict[str, Any] = Field(..., description="A dictionary of input values (int, float, or string).")
    
    class Config:
        extra = Extra.forbid  # Forbid unexpected fields at the top level

# -------------------------- 
# Global Variables & Metrics 
# --------------------------

# Global model reference (loaded during startup)
model = None
label_encoder = None
mse = None
r2 = None

# -------------------------
# Utility: Model Prediction
# -------------------------
def model_predict(model, inputs: Dict[str, Any]) -> Any:
    """
    Prediction function. Replace this with actual model inference logic.
    """
    try:
        # Extract inputs
        position = inputs.get("Position")
        level = inputs.get("Level")

        # Validate inputs
        if position is None or level is None:
            raise ValueError("Both 'Position' and 'Level' must be provided in the input.")

        # Encode categorical input 'Position' if label_encoder exists
        if label_encoder is not None:
            position_encoded = label_encoder.transform([position])[0]
        else:
            raise ValueError("Label encoder not loaded.")
        
        # Convert Level to integer
        try:
            level = int(level)
        except ValueError:
            raise ValueError("Level must be an integer.")

        # Prepare input for prediction
        input_data = [[position_encoded, level]]

        # Make prediction
        prediction = model.predict(input_data)

        return prediction.tolist()  # Convert numpy array to list for JSON serialization

    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        raise ValueError(f"Error during prediction: {e}")

# ---------------------------
# Startup and Shutdown Events
# ---------------------------
@app.on_event("startup")
def load_model():
    """
    Load the model during startup from a pickle file.
    """
    global model, label_encoder, mse, r2
    model_path = "finalized_model.pickle"
    dataset_path = "Position_Salaries.csv"

    try:
        # Load the model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")

        # Load the dataset to fit LabelEncoder
        d = pd.read_csv(dataset_path)

        # Initialize LabelEncoder and fit it
        label_encoder = LabelEncoder()
        d["Position"] = label_encoder.fit_transform(d["Position"])
        logger.info("Label encoder fitted successfully.")

        # Set evaluation metrics
        mse = 462500000.0
        r2 = 0.48611111111111116


    except Exception as e:
        logger.error(f"Failed to load model/data: {e}")
        model = None
        label_encoder = None
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or data failed to load. The service is unavailable."
        )


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down the ML API server.")


# -----------------------
# Exception Handlers
# -----------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An internal error occurred. Please try again later."},
    )


# -------------------
# API Route Handlers
# -------------------

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint. Returns a simple status message to confirm the API is running and the model is loaded.
    """
    if model is None:
          raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later."
        )
    return {"status": "ok"}


@app.post("/predict", tags=["Prediction"])
async def predict(payload: PredictionRequest):
    """
    Prediction endpoint.
    Expects a JSON payload with a dictionary of inputs, which may include strings, integers, or floats.
    Returns model predictions or processed results.
    
    Example Input: 
    {
        "inputs": {
            "Position": "Business Analyst",
            "Level": 1
        }
    }
    """
    global model

    if model is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later."
        )

    try:
        # Call the model_predict function which should contain your actual inference logic.
        predictions = model_predict(model, payload.inputs)
    except ValueError as e:
        logger.error(f"Error during model prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction."
        )

    return {"predictions": predictions}


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """
    Metrics endpoint.
    Returns model evaluation metrics (accuracy, classification matrix) only if available.
    """
    global mse, r2

    if mse is not None and r2 is not None:
        return {
            "MSE": mse,
            "R2": r2
        }
    else:
        return {"message": "No metrics available"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
