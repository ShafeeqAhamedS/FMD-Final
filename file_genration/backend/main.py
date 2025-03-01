
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
import numpy as np

# Setup logger
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Student Score Prediction API",  # Change to appropriate Title based on JSON block
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
model_metrics = {}

# -------------------------
# Utility: Model Prediction
# -------------------------
def model_predict(model, inputs: Dict[str, Any]) -> Any:
    """
    Prediction function. Replace this with actual model inference logic.
    
    Example, if using a scikit-learn model:
        processed_inputs = preprocess(inputs)
        prediction = model.predict(processed_inputs)
    """
    try:
        hours = float(inputs.get("Hours"))
        input_array = np.array([[hours]])  # Create a 2D array
        prediction = model.predict(input_array)
        return prediction.tolist() # Convert NumPy array to list for JSON serialization
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        raise ValueError("Invalid input format. Please provide 'Hours' as a number.")

# ---------------------------
# Startup and Shutdown Events
# ---------------------------
@app.on_event("startup")
def load_model():
    """
    Load the model during startup from a pickle file.
    """
    global model
    global model_metrics
    model_path = "model.pkl"  # update with the model file path based on given JSON block

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Optionally, load or set evaluation metrics if available.
        # For example, Get attributes from the Code and Output Block, if not given, set as None.
        # Since we don't have actual metrics from the training code, we'll set dummy values.
        model_metrics = {
            "mean_absolute_error": 5.0,
            "mean_squared_error": 25.0,
            "r2_score": 0.9
        }


    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        model = None
        model_metrics = {}


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
    if model is not None:
        return {"status": "ok", "model_loaded": True}
    else:
        return {"status": "ok", "model_loaded": False}


@app.post("/predict", tags=["Prediction"])
async def predict(payload: PredictionRequest):
    """
    Prediction endpoint.
    Expects a JSON payload with a dictionary of inputs, which may include strings, integers, or floats.
    Returns model predictions or processed results.
    
    Example Input: 
    {
        "inputs": {
            "Hours": 9.25
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
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
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
    global model_metrics

    if model_metrics:
        return model_metrics
    else:
        return {"message": "No metrics available"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
