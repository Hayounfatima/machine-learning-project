import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_clean_data, scale_features
from models import train_decision_tree, train_logistic_regression
from evaluation import evaluate_model
from visualization import plot_departure_reasons, plot_correlation_heatmap

# Charger et préparer les données
df = load_and_clean_data("HR_Analytics.csv")

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaling pour Logistic Regression
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# Entraînement
dt_model = train_decision_tree(X_train, y_train)
lr_model = train_logistic_regression(X_train_scaled, y_train)

# Évaluation
evaluate_model(dt_model, X_test, y_test, "Decision Tree")
evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")

# Visualisations
plot_departure_reasons(df)
plot_correlation_heatmap(df)