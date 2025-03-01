You are provided with a **React frontend template** (`App.jsx`) that will interface with a backend machine learning model. This template is already designed with error handling, logging, and metrics tracking. You will need to make the following updates to consolidate everything into a production-ready application:

1. **Title and Description:**
   - Replace the placeholders for the title and description of the prediction model. Use a relevant title like `"Prediction Model"` and a description, such as:  
     `"Make predictions based on the given input features."`  
   - Update these sections in the `App.jsx` file to reflect the specifics of the provided model.

2. **Input Handling:**
   - **The form should have all inputs in one object**.
   - Each value in the `inputs` dictionary should accept the appropriate data type (e.g., string, float, int) based on the input variables used in the provided code block.
   - The **inputs object should be structured properly** based on the model’s inference logic. For example:
     - In a regression model, one input might be `input1` (datatype).
     - In a classification model, inputs could be any number of features (e.g., `feature1`, `feature2`).
   - In the prediction logic (in the `handleSubmit` function), **convert the input values** to their relevant data types (string, int, or float) before sending them to the backend API for prediction.

3. **Production-Ready Code:**
   - **Error Handling**:  
     Ensure proper error handling with `try/except` and `HTTPExceptions` in the backend API interaction. Log errors in the frontend and display appropriate error messages for users when inputs are invalid or the backend API is unavailable.
   
   - **Loading State**:  
     During the prediction request (`/predict`), manage the `loading` state. Show a loading spinner when the request is in progress.

   - **Metrics**:  
     Ensure that **model metrics** (such as accuracy, R-squared score, etc., depending on the model type) are displayed correctly if the API provides this data. Implement UI elements to show the metrics clearly. Metrics may vary based on whether the model is a regression or classification model.

4. **Integration of Code Block and Outputs from the Jupyter Notebook:**
   - From the provided **code block** in the JSON (Jupyter notebook), **extract relevant parts** that involve:
     - Input features (e.g., `feature1`, `feature2`, or `input1` depending on the model).
     - Model inference logic for prediction (e.g., `model.predict()` or `regressor.predict()`).
   - **Exclude irrelevant sections** such as those related to data loading, training, or graphing that don't pertain to the prediction task.
   - The **inputs should match** the model’s required format. For example:
     - `input1: input1 (datatype)` for regression.
     - `input1: feature1 (int), input2: feature2 (float)` for classification.
   - The backend model should accept the **inputs** in a proper format and return the predicted value or class.

5. **Backend Interaction for Prediction (`/predict` endpoint):**
   - When submitting the form, send the `inputs` object to the backend API endpoint (`/predict`).
   - **Do not change the template code** anywhere other than where you are required to integrate the new logic. Only enhance existing code and add the necessary functionality for input validation and prediction.

6. **Final Adjustments:**
   - **Clear Error Messages**: Display clear error messages if there’s a failure in input conversion or if the backend API is down.
   - **Input Validation**: Ensure input fields are validated (e.g., numeric fields should only accept valid numbers, categorical fields should be checked for valid categories).
   - **Model Accuracy**: If available, display the model accuracy, R-squared score, or any other relevant metric returned by the backend API.

### Backend Request Response Models

#### **Route: `/health`**
- **Request Model:** None
- **Response Model:**
  ```json
  {
    "status": "ok"
  }
  ```

---

#### **Route: `/predict`**
- **Request Model:**
  ```json
  {
    "inputs": {
      "<input_key>": <value>
    }
  }
  ```
  - Example: `{"inputs": {"feature1": 5, "feature2": 7.5}}` (for classification or regression inputs)

- **Response Model:**
  ```json
  {
    "predictions": [<predicted_value_or_class>]
  }
  ```
  - Example for regression: `{"predictions": [85.0]}`
  - Example for classification: `{"predictions": ["class_A"]}`

---

#### **Route: `/metrics`**
- **Request Model:** None
- **Response Model:**
  ```json
  {
    "accuracy": <float>,         // Accuracy for classification tasks
    "r2_score": <float>,         // R-squared score for regression tasks
    "classification_matrix": <null or matrix>, // Confusion matrix for classification models
    "precision": <float>,        // Precision for classification tasks
    "recall": <float>,           // Recall for classification tasks
  }
  ```

### Example Expected Output:

- When the form is submitted:
  - The `inputs` object is sent to the backend API with the appropriate data types.
  - The backend API returns a prediction, which is displayed on the frontend as: `"Predicted Score: 85"` (regression) or `"Predicted Class: class_A"` (classification).
  - Model metrics (e.g., accuracy, R-squared score) are shown under "Model Metrics."
  - If the backend is not reachable, display a clear error message such as: `"Prediction failed. Please check your input and try again."`