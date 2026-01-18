import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path="../data/cleaned_data.csv"):
    df = pd.read_csv(path)
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