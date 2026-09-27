import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def run():
    with open("data_preprocessed.pkl", "rb") as f:
        data = pickle.load(f)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    print("\n=== Comparaison avec sklearn ===")

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    print("Decision Tree (sklearn)")
    print(" Accuracy:", accuracy_score(y_test, y_pred_dt))
    print(" F1-score:", f1_score(y_test, y_pred_dt))

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print("\nLogistic Regression (sklearn)")
    print(" Accuracy:", accuracy_score(y_test, y_pred_lr))
    print(" F1-score:", f1_score(y_test, y_pred_lr))

