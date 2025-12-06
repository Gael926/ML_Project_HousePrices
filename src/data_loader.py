import pandas as pd
import os

def load_data(train_path, test_path):
    # Loads train and test datasets from CSV files.
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Data loaded successfully.")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    
    return df_train, df_test
