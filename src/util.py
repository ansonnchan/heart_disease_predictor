import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importances(model, feature_names):
    # Extract feature importance scores from tree-based model
    # Higher values indicate features that contribute more to predictions
    importances = pd.Series(model.feature_importances_, index=feature_names)
    
    importances.sort_values(ascending=False).plot(kind='bar', figsize=(12,4))
    plt.title("Feature Importances")
    plt.show()