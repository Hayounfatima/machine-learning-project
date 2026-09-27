import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Supprimer colonnes inutiles
    cols_to_drop = ['EmpID','EmployeeCount','Over18','StandardHours','EmployeeNumber']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Remplir valeurs manquantes
    if 'YearsWithCurrManager' in df.columns:
        df['YearsWithCurrManager'] = df['YearsWithCurrManager'].fillna(df['YearsWithCurrManager'].median())

    # Encodage des variables catégorielles
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled