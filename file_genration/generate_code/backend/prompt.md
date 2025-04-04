**Objective:**  
We need to create a production-ready FastAPI service for a machine learning (ML) model. The service should provide endpoints for the following functionalities:

1. **Health Check Endpoint:**  
   An endpoint `/health` to verify the API is running and the model is loaded.

2. **Prediction Endpoint:**  
   An endpoint `/predict` that accepts a JSON payload with an "inputs" object.  
   Each value in the "inputs" dictionary should accept strings, floats, or integers, as specified in the attached JSON code block.  
   - The prediction logic should convert the inputs into appropriate data types before invoking the model.
   - Categorical Inputs:
If the input involves categorical values, the string value will be provided and the model should handle the encoding internally based on encoding logic provided in the JSON code block.
   - The prediction response should include the output, which could be a result of processing, transformed data, or model inference.  

3. **Metrics Endpoint:**  
   The `/metrics` endpoint should return model evaluation metrics (accuracy, confusion matrix, R1 score, etc.) only if they are available.  
   The metrics such as accuracy and classification matrix are provided in the JSON block, include them in the response.
   If no metrics are provided, indicate the lack of metrics, the API should return "No metrics available".
   Do not write code to perform the training at any cost

5. **Error Handling and Logging:**  
   - Proper error handling using `try/except` blocks.
   - Log all errors appropriately.
   - Ensure that the API returns clear error messages if the model is not loaded or if input conversion fails.

6. **Startup and Shutdown Events:**  
   The model should be loaded from a pickle file during startup. If the model fails to load, return a service unavailable status.
   
7. **Dynamic Metrics:**  
   If evaluation metrics such as accuracy and classification matrix are provided in the JSON block, include them in the response. If no metrics are provided, indicate the lack of metrics.
