### Prompt

### Prompt

**Objective:**  
You are an AI code generator. Your task is to generate a Python script (`model.py`) that loads a saved machine learning model, preprocesses input data, runs inference, and evaluates the model's performance. The script should be production-ready, modular, and follow best practices for error handling, logging, and code structure.

**Inputs Provided:**  
1. **Template Python File**: A skeleton script with placeholders for key functionalities.  
2. **Dataset Preview**: A preview of the dataset (column names and sample rows).  
3. **Code Blocks JSON**: JSON containing Python code blocks used during training (e.g., preprocessing, model training, evaluation).  

**Requirements:**  
The script must:  
1. **Load the Dataset**: Load the dataset from a CSV file and validate its structure. Always Add './' while reading file
2. **Preprocess Data**: Apply necessary preprocessing steps (e.g., encoding, scaling, feature engineering) based on the training pipeline.  
3. **Load the Model**: Load a pre-trained model from a file (e.g., `.pkl`, `.joblib`).  Always Add './' while loading model
4. **Run Inference**: Select a sample row from the dataset and generate predictions using the loaded model.  
5. **Evaluate the Model**: Compute evaluation metrics (e.g., accuracy, F1-score for classification; MSE, R² for regression).  
6. **Error Handling**: Include robust exception handling for missing data, incorrect data types, or model loading issues.  
7. **Logging**: Use logging to track the script's execution and report errors or important events.  
8. **Modularity**: Break the code into reusable functions (e.g., `load_dataset`, `preprocess_data`, `load_model`, `model_inference`, `model_evaluation`).  
9. **Generalization**: Avoid hardcoding column names, file paths, or model-specific details. Use variables and configurations that can be easily modified.  

**Expected Output:**  
A well-structured Python script (`model.py`) that implements the above functionality. The script should be clear, well-commented, and ready for production use.

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

 None#### 3. Template Code

Template: model.py
This script:
✅ Loads the dataset preview and extracts one row
✅ Detects preprocessing steps (scaling, encoding, feature extraction)
✅ Loads the saved model and runs inference
✅ Evaluates model performance
Change the file names, paths, and model loading based on the given code blocks.  
Change the preprocessing steps based on the data types and code blocks.  
Change the evaluation metrics based on the target variable type.  
Do not hardcode column names or model paths.  
Do not include training code or model saving.  

```py
# Import necessary libraries (pandas, numpy, scikit-learn, etc, based on given code blocks)
import joblib  # To load the trained model
import logging # To log messages
from sklearn.model_selection import train_test_split # To split the dataset into training and testing sets

# Ignore sklearn warnings
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load dataset (Replace 'dataset.csv' with actual dataset file if required)
def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        exit()

def preprocess_data(df):
    """Apply preprocessing steps like scaling and encoding based on the data types and code blocks, do only the necessary steps as per the given code blocks."""
    try:
        # Encoding
        logging.info("encoded successfully.")
        
        # Scaling
        logging.info("Features scaled successfully.")
        
        # Feature Engineering
        logging.info("Feature engineering done.")
        
        
        return df
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        exit()

def load_model(model_path):
    """Load the pre-trained model from a file."""
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

def model_inference(model, sample_row):
    """Make prediction using the trained model."""
    try:
        prediction = model.predict(sample_row)
        logging.info(f"Prediction made successfully: {prediction}")
        return prediction
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        exit()

def model_evaluation(model, X_test, y_test):
    """Evaluate model performance using MSE and R2."""
    try:
        # Predicting the target variable
        y_pred = model.predict(X_test)
        
        if y.dtype == "object":  # Classification case (if target is categorical) (Use all classification metrics)
            # Use all 5 classification metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(y, y_pred, average='weighted')
            cm = confusion_matrix(y, y_pred)
            
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1-Score: {f1}")
            logging.info(f"Confusion Matrix:\n{cm}")
        else:  # Regression case
            # Calculate regression metrics, use all regression metrics
            # Use all 5 regression metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / y)) * 100

            # Log and print the regression evaluation results
            logging.info(f"MAE: {mae}")
            logging.info(f"MSE: {mse}")
            logging.info(f"RMSE: {rmse}")
            logging.info(f"R²: {r2}")
            logging.info(f"MAPE: {mape}")
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        exit()

def main():
    # Load dataset (Replace 'dataset.csv' with actual dataset file if required)
    df = load_dataset("./dataset.csv") # Always load dataset from the same directory as the script ./

    # Detect numerical and categorical columns for preprocessing based on given code blocks
    # Preprocess data
    df = preprocess_data(df)
    
    # Prepare features and target variable
    # Detect target column and features based on given code blocks
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Load trained model (Replace 'model.pkl' with actual trained model file based on given code blocks)
    model = load_model("model.pkl")  # Always load model from the same directory as the script ./
    
    # Run model inference on a sample row
    sample_row = X_test.sample(1)  # Select one row for prediction based on given code blocks
    prediction = model_inference(model, sample_row)
    
    # Evaluate the model (All evaluation metrics based on the target variable type)
    model_evaluation(model, X_test, y_test)
    
    # Print results
    logging.info(f"Sample Prediction: {prediction}")
    logging.info(f"Model Evaluation - MSE: {mse}, R2: {r2}")

if __name__ == "__main__":
    main()
```