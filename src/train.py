from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X_train, y_train, n_estimators=200, min_samples_leaf=5,
                        class_weight='balanced', random_state=42):
    # Initialize Random Forest with specified hyperparameters
    rf = RandomForestClassifier(
        n_estimators=n_estimators,        # Number of decision trees
        min_samples_leaf=min_samples_leaf,  # Prevents overfitting on small leaves
        class_weight=class_weight,          # Handles class imbalance
        random_state=random_state           # Ensures reproducible results
    )
    
    # Train model on the provided data
    rf.fit(X_train, y_train)
    return rf

def save_model(model, path="../models/best_model.pkl"):
    # Save trained model to disk for future use
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path="../models/best_model.pkl"):
    model = joblib.load(path)
    return model
