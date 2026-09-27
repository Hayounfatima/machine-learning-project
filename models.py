from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    return dt

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    return lr