import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df_train, df_test):
    # Cleans and preprocesses the training and test data.
    df_train_len = len(df_train)
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Drop Id
    if 'Id' in df_all.columns:
        df_all = df_all.drop('Id', axis=1)

    # Handling Missing Values
    # Categorical with "None" meaning
    cols_fillna_none = ["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu", 
                        "GarageType", "GarageFinish", "GarageQual", "GarageCond", 
                        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    for col in cols_fillna_none:
        if col in df_all.columns:
            df_all[col] = df_all[col].fillna("None")

    # Numerical with 0 meaning
    cols_fillna_0 = ["GarageYrBlt", "GarageArea", "GarageCars", "MasVnrArea", 
                     "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
    for col in cols_fillna_0:
        if col in df_all.columns:
            df_all[col] = df_all[col].fillna(0)

    # LotFrontage: Median by Neighborhood
    df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    # Mode imputation for remaining
    cols_fillna_mode = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType", "Utilities", "Functional"]
    for col in cols_fillna_mode:
        if col in df_all.columns:
            df_all[col] = df_all[col].fillna(df_all[col].mode()[0])
    
    # Feature Engineering
    df_all['TotalSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']

    # Type Conversion
    df_all['MSSubClass'] = df_all['MSSubClass'].apply(str)

    # Split back
    X_train_processed = df_all.iloc[:df_train_len, :]
    X_test_processed = df_all.iloc[df_train_len:, :]

    # Handle Target
    # Log transform SalePrice in train
    y_train = df_train['SalePrice']
    y_train = np.log1p(y_train)
    
    # Drop SalePrice from X_train if it leaked back
    if 'SalePrice' in X_train_processed.columns:
        X_train_processed = X_train_processed.drop('SalePrice', axis=1)
    if 'SalePrice' in X_test_processed.columns:
        X_test_processed = X_test_processed.drop('SalePrice', axis=1)

    return X_train_processed, y_train, X_test_processed

def get_preprocessor(X_train):
    # Returns the ColumnTransformer for preprocessing.
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor
