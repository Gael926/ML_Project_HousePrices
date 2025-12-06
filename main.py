import pandas as pd
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH
from src.data_loader import load_data
from src.preprocessing import preprocess_data, get_preprocessor
from src.training import train_linear_regression, train_random_forest, train_xgboost, train_final_xgboost

def main():
    print("Starting House Prices Regression Pipeline")
    
    # Load Data
    print("\nLoading Data")
    df_train, df_test = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    
    # Preprocess Data
    print("\nPreprocessing Data")
    X_train, y_train, X_test = preprocess_data(df_train, df_test)
    
    preprocessor = get_preprocessor(X_train)
    
    # Store results
    results = []
    
    # Linear Regression
    print("\nTraining Linear Regression")
    res_lr = train_linear_regression(X_train, y_train, preprocessor)
    results.append(res_lr)
    
    # Random Forest
    print("\nTraining Random Forest")
    res_rf_base, res_rf_tuned = train_random_forest(X_train, y_train, preprocessor)
    results.append(res_rf_base)
    results.append(res_rf_tuned)
    
    # XGBoost Baseline
    print("\nTraining XGBoost Baseline")
    res_xgb_base = train_xgboost(X_train, y_train, preprocessor)
    results.append(res_xgb_base)
    
    # Final XGBoost
    print("\nTraining Final XGBoost (Early Stopping)")
    res_xgb_final = train_final_xgboost(X_train, y_train, preprocessor)
    results.append(res_xgb_final)
    
    # Summary
    print("\n\nFINAL RESULTS TABLE")
    results_df = pd.DataFrame(results)
    print(results_df)
    
    print("\nPipeline completed successfully")

if __name__ == "__main__":
    main()
