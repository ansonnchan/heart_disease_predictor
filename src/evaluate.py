"""
Evaluation module
Contains functions to evaluate classification models:
ROC-AUC, accuracy, confusion matrix, and classification report.

ROC-AUC measures how well a model can distinguish between the positive 
and negative classes across all possible thresholds.A high ROC-AUC 
(close to 1.0) means the model ranks positive cases higher than negative ones
 very effectively, while 0.5 means it performs no better than random guessing.
"""

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Get predicted probabilities for positive class (heart disease)
    # Using predict_proba instead of predict for threshold flexibility
    y_pred_proba = model.predict_proba(X_test)[:,1]
    
    # Apply custom threshold (default 0.5) to convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics - ROC-AUC for probability ranking, accuracy for classification
    roc_auc = roc_auc_score(y_test, y_pred_proba)  # Area under ROC curve (0.5-1.0)
    acc = accuracy_score(y_test, y_pred)           # Simple accuracy score
    cm = confusion_matrix(y_test, y_pred)          # TP, FP, FN, TN counts
    report = classification_report(y_test, y_pred) # Precision, recall, F1 per class
    

    print("ROC-AUC:", roc_auc)
    print("Accuracy:", acc)
    print("Classification Report:\n", report)
    

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    return roc_auc, acc, cm, report