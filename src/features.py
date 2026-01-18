def feature_engineering(df):
    # Combine blood pressure and hypertension flag into risk score
    df['bp_risk'] = df['blood_pressure'] * (1 + df['high_blood_pressure'])
    
    # Combine sugar consumption and diabetes into one feature 
    df['sugar_diabetes'] = df['sugar_consumption'] * df['diabetes']
    
    # Remove original columns to prevent multicollinearity
    df = df.drop(columns=['blood_pressure', 'high_blood_pressure',
                          'sugar_consumption', 'diabetes'])
    return df