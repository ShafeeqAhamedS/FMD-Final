### Prompt

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

#### 1. Dataset Preview

 Hours  Scores
   2.5      21
   5.1      47
   ...     ...
   6.9      76
   7.8      86

#### 2. Code Blocks JSON

[
    {
        "source": [
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.linear_model import LinearRegression"
        ],
        "outputs": []
    },
    {
        "source": [
            "df = pd.read_csv('student_scores - student_scores.csv')\n",
            "df.head()"
        ],
        "outputs": [
            [
                "   Hours  Scores\n",
                "0    2.5      21\n",
                "1    5.1      47\n",
                "2    3.2      27\n",
                "3    8.5      75\n",
                "4    3.5      30"
            ]
        ]
    },
    {
        "source": [
            "df.tail()"
        ],
        "outputs": [
            [
                "    Hours  Scores\n",
                "20    2.7      30\n",
                "21    4.8      54\n",
                "22    3.8      35\n",
                "23    6.9      76\n",
                "24    7.8      86"
            ]
        ]
    },
    {
        "source": [
            "x=df.iloc[:,:-1].values\n",
            "y=df.iloc[:,1].values\n",
            "print(x)\n",
            "print(y)"
        ],
        "outputs": []
    },
    {
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state=0)\n",
            "regressor = LinearRegression()\n",
            "regressor.fit(x_train,y_train)\n",
            "y_pred=regressor.predict(x_test)"
        ],
        "outputs": []
    },
    {
        "source": [
            "#for training data\n",
            "plt.scatter(x_train,y_train,color = \"black\")\n",
            "plt.plot(x_train,regressor.predict(x_train),color= \"lightblue\")\n",
            "plt.title(\"hours Vs scores(train)\")\n",
            "plt.xlabel(\"hours\")\n",
            "plt.ylabel(\"scores\")\n",
            "plt.show()"
        ],
        "outputs": [
            [
                "<Figure size 640x480 with 1 Axes>"
            ]
        ]
    },
    {
        "source": [
            "#for test data\n",
            "plt.scatter(x_test,y_test,color='#FF6766')\n",
            "##FF6766 is light red Color\n",
            "plt.plot(x_test,regressor.predict(x_test),color='#D099E2')\n",
            "##D099E2 is Lavendar Color\n",
            "plt.title(\"hours Vs scores(test)\")\n",
            "plt.xlabel(\"hours\")\n",
            "plt.ylabel(\"scores\")\n",
            "plt.show()"
        ],
        "outputs": [
            [
                "<Figure size 640x480 with 1 Axes>"
            ]
        ]
    },
    {
        "source": [
            "# Save model as pickle\n",
            "import pickle\n",
            "pickle.dump(regressor,open('model.pkl','wb'))"
        ],
        "outputs": []
    }
] 

 INFO:root:Dataset loaded successfully.
INFO:root:No explicit encoding or scaling required based on the provided code blocks.
INFO:root:Model loaded successfully.
INFO:root:Prediction made successfully: [17.04289179]
ERROR:root:Error in model evaluation: 'numpy.ndarray' object has no attribute 'iloc'
#### 3. Template Code

### **Provided Base Template (`App.jsx`)**:

```js
import { useState, useEffect } from "react"
import axios from "axios"

import {API_BASE_URL} from "./api_route"

const App = () => {
  const [inputs, setInputs] = useState({ feature1: "", feature2: "" })  // Adjusted based on the model inputs from JSON Code block
  const [prediction, setPrediction] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [healthStatus, setHealthStatus] = useState("Checking...")
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    checkHealth()
    fetchMetrics()
  }, [])

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`)
      setHealthStatus(response.data.status === "ok" ? "Online" : "Offline")
    } catch (err) {
      setHealthStatus("Offline")
    }
  }

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/metrics`)
      setMetrics(response.data)
    } catch (err) {
      setMetrics(null)
    }
  }

  const handleInputChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setPrediction(null)
    setLoading(true)

    // Convert inputs to the appropriate data types (float, int, etc.)
    const convertedInputs = {
      feature1: parseFloat(inputs.feature1),
      feature2: parseFloat(inputs.feature2),  // only If applicable
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, { inputs: convertedInputs })
      setPrediction(response.data.predictions)
    } catch (err) {
      setError("Prediction failed. Please check your input and try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-indigo-100 py-6 sm:py-8 md:py-10 lg:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-xs sm:max-w-md md:max-w-lg lg:max-w-xl xl:max-w-2xl mx-auto bg-white rounded-xl shadow-md overflow-hidden">
        <div className="p-6 sm:p-8 md:p-10">
          <div className="uppercase tracking-wide text-sm text-indigo-500 font-semibold mb-1">ML Model Prediction</div>
          <h1 className="text-lg sm:text-xl md:text-2xl lg:text-3xl font-medium text-black mb-2">
            Prediction Model {/* Update the title */}
          </h1>
          <p className="mt-2 text-sm sm:text-base text-gray-500">
            Make predictions based on the given input features. {/* Update description */}
          </p>

          <div className="mt-4 flex items-center">
            <span className="text-sm sm:text-base text-gray-700 mr-2">API Status:</span>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs sm:text-sm font-medium ${
                healthStatus === "Online" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
              }`}
            >
              {healthStatus}
            </span>
          </div>

          <form onSubmit={handleSubmit} className="mt-6">
            <label className="block">
              <span className="text-sm sm:text-base text-gray-700">Feature1: </span>
              <input
                type="number"
                name="feature1"
                value={inputs.feature1}
                onChange={handleInputChange}
                className="rounded-md mt-1 block w-full px-3 py-2 border border-gray-300 shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo"
                required
              />
            </label>

            <label className="block mt-4">
              <span className="text-sm sm:text-base text-gray-700">Feature2: </span>
              <input
                type="number"
                name="feature2"
                value={inputs.feature2}
                onChange={handleInputChange}
                className="rounded-md mt-1 block w-full px-3 py-2 border border-gray-300 shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo"
                required
              />
            </label>

            <button
              type="submit"
              className="mt-4 w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50 transition duration-150 ease-in-out text-sm sm:text-base"
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Predicting...
                </span>
              ) : (
                "Predict"
              )}
            </button>
          </form>

          {prediction && (
            <div className="mt-6">
              <p className="text-xl sm:text-2xl font-semibold text-gray-800">
                Prediction: {prediction[0]} {/* Update based on response & response model */}
              </p>
            </div>
          )}

          {/*Use the below metrics as it is*/}

          {metrics && Object.keys(metrics).length > 0 && (
            <div className="mt-8">
              <h2 className="text-base sm:text-lg md:text-xl font-semibold text-gray-800 mb-4">
                Model Metrics
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {Object.entries(metrics).map(([key, value]) => (
                  value && (
                    <div key={key} className="bg-indigo-100 p-4 rounded-md">
                      <h3 className="text-xs sm:text-sm font-medium text-indigo-800">{key}</h3>
                      <p className="mt-1 text-lg sm:text-xl md:text-2xl font-semibold text-indigo-900">
                        {typeof value === "number" ? value.toFixed(4) : value}
                      </p>
                    </div>
                  )
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
```