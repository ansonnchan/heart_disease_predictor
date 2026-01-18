import os
from preprocessing import load_data, oversample_smote, split_features_target, scale_features
from features import feature_engineering
from train import train_random_forest
from evaluate import evaluate_model    
from sklearn.model_selection import train_test_split


def main():
    """
    Main function to run the Heart Disease Prediction pipeline:
    1. Load raw data
    2. Preprocess and clean data
    3. Generate features
    4. Train Random Forest model
    5. Evaluate model performance
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root folder
    raw_data_path = os.path.join(project_root, "data", "heart_disease_raw.csv")
    

    df = load_data(raw_data_path)
    
    X, y = split_features_target(df)
    X = feature_engineering(X)  

    X_res, y_res = oversample_smote(X, y)


    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)


    model = train_random_forest(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
