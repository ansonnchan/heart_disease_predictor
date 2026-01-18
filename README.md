# Heart Disease Predictor

An introductory **machine learning classification project** that predicts the likelihood of heart disease based on patient health data. This project demonstrates data cleaning, feature engineering, model training, evaluation, and deployment.

## Dataset

The dataset used is the **Heart Disease dataset** from Kaggle:  [https://www.kaggle.com/datasets/oktayrdeki/heart-disease](https://www.kaggle.com/datasets/oktayrdeki/heart-disease)  

- Contains 10,000 patient records with 21 features including age, gender, blood pressure, cholesterol levels, lifestyle factors, and heart disease status.  
- The target variable is `heart_disease_status` (Yes/No).


## Tools and Libraries

- **Python 3.x**  
- **Data manipulation:** `pandas`, `numpy`  
- **Visualization:** `matplotlib`, `seaborn`  
- **Machine learning:** `scikit-learn`, `imblearn` (SMOTE)  
- **Model persistence:** `joblib`  

## Data Cleaning and Preprocessing

1. Standardized column names to lowercase with underscores.  
2. Filled missing numeric values using the column median.  
3. Encoded binary columns (Yes/No) as 1/0.  
4. Converted ordinal features (Low/Medium/High) into numerical values.  
5. Encoded gender as 1 (Male) / 0 (Female).  
6. Dropped `alcohol_consumption` due to excessive missing values (~25%).  
7. Removed any remaining rows with missing values to ensure a complete, clean dataset for modeling.

## Feature Engineering

- Combined `blood_pressure` and `high_blood_pressure` into a new feature `bp_risk`.  
- Combined `sugar_consumption` and `diabetes` into `sugar_diabetes`.  
- Original columns were removed after combination to prevent multicollinearity.

## Model Training

- **Model used:** Random Forest Classifier  
- **Class imbalance handled:** Synthetic Minority Oversampling Technique (SMOTE)  
- **Train/Test split:** 80/20%  
- **Feature scaling:** StandardScaler  
- **Hyperparameters:**  
  - `n_estimators=200`  
  - `min_samples_leaf=5`  
  - `class_weight='balanced'`  

## Evaluation

- **Metrics used:**  
  - Receiver Operating Characteristic Area Under the Curve (ROC-AUC)  
  - Accuracy  
  - Precision, Recall, F1-score  
  - Confusion matrix visualization  

- **Example Results on Test Set:**  
  - ROC-AUC: 0.8916643021921764
  - Accuracy: 0.8634177621032382
  - Class 0 (No Heart Disease) – Precision: 0.78, Recall: 1.00, F1-score: 0.88  
  - Class 1 (Heart Disease) – Precision: 1.00, Recall: 0.73, F1-score: 0.84  

- **Confusion Matrix:**  

  ![Alt Text](https://github.com/ansonnchan/heart_disease_predictor/blob/main/results/confusion_matrix.png)


## Results

- Model is able to **distinguish patients with heart disease effectively**, as shown by ROC-AUC of 0.89.  
- High precision for heart disease predictions ensures minimal false positives.  
- Good recall for healthy patients ensures most non-heart disease patients are correctly identified.  


## Deployment

- Trained model is saved as `best_model.pkl` for reuse.  
- Can be loaded with `joblib` for predictions on new patient data.  
- Future work can include integrating this model into a web app or API for real-time predictions.


## Usage

1. Clone the repository:  
```bash
git clone https://github.com/ansonnchan/heart_disease_predictor
```

2. Make sure you have the required imports (see above)
3. 
4. Run the pipeline:

Run `python src/main.py`

5. Load saved model for new predictions:
```bash
import joblib
model = joblib.load("models/best_model.pkl")
predictions = model.predict(new_patient_data)
```
