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