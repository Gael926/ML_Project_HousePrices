import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

def evaluate_regression(y_true, y_pred, model_name="Model"):
    # Evaluates regression model performance and returns a dictionary of metrics.
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"Model: {model_name}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MedAE: {medae:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}\n")

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "MAPE": mape,
        "R2": r2
    }
