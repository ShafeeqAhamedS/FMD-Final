
# Import necessary libraries (pandas, numpy, scikit-learn, etc, based on given code blocks)
import pandas as pd
import numpy as np
import joblib  # To load the trained model
import logging # To log messages
from sklearn.model_selection import train_test_split # To split the dataset into training and testing sets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
        # No encoding or scaling needed based on the provided code blocks
        # The code blocks directly use the numerical data in the CSV
        logging.info("No explicit encoding or scaling required based on the provided code blocks.")
        return df
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        exit()

def load_model(model_path):
    """Load the pre-trained model from a file."""
    try:
        import pickle
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
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
        
        if isinstance(y_test.iloc[0], str) or isinstance(y_test.iloc[0], object):  # Classification case (if target is categorical) (Use all classification metrics)
            # Use all 5 classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1-Score: {f1}")
            logging.info(f"Confusion Matrix:\n{cm}")
            
            mae = None
            mse = None
            rmse = None
            r2 = None
            mape = None
        else:  # Regression case
            # Calculate regression metrics, use all regression metrics
            # Use all 5 regression metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            # Log and print the regression evaluation results
            logging.info(f"MAE: {mae}")
            logging.info(f"MSE: {mse}")
            logging.info(f"RMSE: {rmse}")
            logging.info(f"R²: {r2}")
            logging.info(f"MAPE: {mape}")
            
            accuracy = None
            precision = None
            recall = None
            f1 = None
            cm = None
            
        return mae, mse, rmse, r2, mape, accuracy, precision, recall, f1, cm
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        exit()

def main():
    # Load dataset (Replace 'dataset.csv' with actual dataset file if required)
    file_path = "./student_scores - student_scores.csv"
    df = load_dataset(file_path) # Always load dataset from the same directory as the script ./

    # Detect numerical and categorical columns for preprocessing based on given code blocks
    # Preprocess data
    df = preprocess_data(df)
    
    # Prepare features and target variable
    # Detect target column and features based on given code blocks
    # X=df.iloc[:,:-1].values
    # y=df.iloc[:,1].values
    X = df[['Hours']].values  # Features (independent variables)
    y = df['Scores'].values   # Target (dependent variable)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
    
    # Load trained model (Replace 'model.pkl' with actual trained model file based on given code blocks)
    model_path = "./model.pkl"
    model = load_model(model_path)  # Always load model from the same directory as the script ./
    
    # Run model inference on a sample row
    sample_row = X_test[[0]]  # Select one row for prediction based on given code blocks. Reshape for single sample.
    prediction = model_inference(model, sample_row.reshape(1, -1))
    
    # Evaluate the model (All evaluation metrics based on the target variable type)
    mae, mse, rmse, r2, mape, accuracy, precision, recall, f1, cm = model_evaluation(model, X_test, y_test)
    
    # Print results
    logging.info(f"Sample Prediction: {prediction}")
    logging.info(f"Model Evaluation - MSE: {mse}, R2: {r2}")

if __name__ == "__main__":
    main()
