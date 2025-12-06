import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from src.evaluation import evaluate_regression
from src.config import RANDOM_STATE, FINAL_MODEL_PATH

def train_linear_regression(X_train, y_train, preprocessor):
    # Trains and evaluates a baseline Linear Regression model.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Train test split for internal evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
    
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_val)
    
    return evaluate_regression(y_val, y_pred, "Linear Regression")

def train_random_forest(X_train, y_train, preprocessor):
    # Trains baseline and tuned Random Forest models.

    # Baseline
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    pipeline_base = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ])
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
    
    pipeline_base.fit(X_tr, y_tr)
    y_pred_base = pipeline_base.predict(X_val)
    res_base = evaluate_regression(y_val, y_pred_base, "Random Forest (Baseline)")
    
    # Tuned
    param_dist = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5], 
        'regressor__min_samples_leaf': [1, 2]
    }
    
    random_search = RandomizedSearchCV(
        pipeline_base, 
        param_distributions=param_dist, 
        n_iter=5,
        cv=3, 
        scoring='neg_root_mean_squared_error', 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    random_search.fit(X_tr, y_tr)
    best_rf = random_search.best_estimator_
    y_pred_tuned = best_rf.predict(X_val)
    res_tuned = evaluate_regression(y_val, y_pred_tuned, "Random Forest (Tuned)")
    
    return res_base, res_tuned

def train_xgboost(X_train, y_train, preprocessor):
    # Trains baseline XGBoost.
    pipeline_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            objective='reg:squarederror', 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        ))
    ])
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
    
    pipeline_xgb.fit(X_tr, y_tr)
    y_pred = pipeline_xgb.predict(X_val)
    
    return evaluate_regression(y_val, y_pred, "XGBoost (Baseline)")

def train_final_xgboost(X_train, y_train, preprocessor):
    # Trains the final XGBoost model with early stopping.
    
    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=RANDOM_STATE)
    
    # Preprocess
    X_tr_prep = preprocessor.fit_transform(X_tr)
    X_val_prep = preprocessor.transform(X_val)
    
    # DMatrix
    dtrain = xgb.DMatrix(X_tr_prep, label=y_tr)
    dval = xgb.DMatrix(X_val_prep, label=y_val)
    
    # Params
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": RANDOM_STATE,
        "max_depth": 2,
        "learning_rate": 0.07,
        "subsample": 0.74,
        "colsample_bytree": 0.7,
        "gamma": 0.02,
        "reg_alpha": 0.11,
        "reg_lambda": 0.65
    }
    
    evals_result = {}
    
    final_xgb = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=evals_result
    )
    
    print(f"Final XGBoost Best Iteration: {final_xgb.best_iteration}")
    
    # Save Model
    joblib.dump(final_xgb, FINAL_MODEL_PATH)
    print(f"Model saved to {FINAL_MODEL_PATH}")
    
    # Evaluate
    y_pred_final = final_xgb.predict(dval)
    
    return evaluate_regression(y_val, y_pred_final, "XGBoost (Final)")
