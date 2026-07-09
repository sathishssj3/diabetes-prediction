import os
import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
DATA_PATH = os.path.join("data", "diabetes.csv")

def download_data():
    """Downloads the dataset if it doesn't exist locally."""
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Dataset downloaded.")

def load_data():
    """Loads the dataset into a pandas DataFrame."""
    download_data()
    return pd.read_csv(DATA_PATH)

def preprocess_data(df):
    """
    Cleans the data by handling missing values (zeros) 
    and scales features for training.
    """
    # Some columns have 0 where it's biologically impossible (e.g. BMI, BloodPressure).
    # We replace 0 with NaN, then impute with the median of the column.
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, pd.NA)
    
    # Ensure columns are numeric before computing median
    for col in cols_with_zeros:
        df[col] = pd.to_numeric(df[col])
        df[col] = df[col].fillna(df[col].median())

    # Separate features and target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference
    import joblib
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(scaler, os.path.join('models', 'scaler.joblib'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test
