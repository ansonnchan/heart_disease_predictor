import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

def load_data(path="../data/heart_disease_raw.csv"):
 
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Binary encoding for yes/no columns
    yes_no_cols = [
        "smoking", "family_heart_disease", "diabetes",
        "high_blood_pressure", "low_hdl_cholesterol",
        "high_ldl_cholesterol", "heart_disease_status"
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Ordinal encoding
    ordinal_map = {"low": 0, "medium": 1, "high": 2}
    ordinal_cols = ["exercise_habits", "alcohol_consumption", "stress_level", "sugar_consumption"]
    for col in ordinal_cols:
        if col in df.columns:  # skip if the column was dropped previously
            df[col] = df[col].str.lower().map(ordinal_map)

    # Gender encoding
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # Drop columns with too many missing values if needed
    if "alcohol_consumption" in df.columns:
        df = df.drop(columns=["alcohol_consumption"])

    # Drop any remaining rows with missing values
    df = df.dropna()

    return df


def split_features_target(df, target="heart_disease_status"):
    # Separate features from target variable
    X = df.drop(columns=[target]) 
    y = df[target]
    return X, y


def scale_features(X_train, X_test):
    # Initialize StandardScaler; helps algorithms that are sensitive to feature magnitudes
    scaler = StandardScaler()
    
    #fit on training data to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using the same parameters learned from training
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def oversample_smote(X, y, random_state=42):
    # Initialize SMOTE to synthetically create samples for minority class to balance dataset
    sm = SMOTE(random_state=random_state)

    X_res, y_res = sm.fit_resample(X, y)
    
    return X_res, y_res