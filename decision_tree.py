import pickle
import numpy as np

with open("data_preprocessed.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
y_train = data["y_train"]
X_test  = data["X_test"]
y_test  = data["y_test"]

# ===== Feature selection by low variance =====
variances = X_train.var(axis=0)
threshold = 0.01  # à tester : 0.001, 0.01, 0.05
keep = variances > threshold

print("Features avant:", X_train.shape[1])
print("Features apres:", keep.sum())

X_train = X_train[:, keep]
X_test  = X_test[:, keep]

#foncrion de base gini
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs ** 2)

#trouver la mielleur division(split)
def best_split(X, y):
    n_samples, n_features = X.shape
    best_gini = float("inf")
    best_feature, best_threshold = None, None

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])

        for t in thresholds:
            left = y[X[:, feature] <= t]
            right = y[X[:, feature] > t]

            if len(left) == 0 or len(right) == 0:
                continue

            g = (len(left) / n_samples) * gini(left) + \
                (len(right) / n_samples) * gini(right)

            if g < best_gini:
                best_gini = g
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold

#node de l'arbre +recursive tree build
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

#construction de l'arbre
def build_tree(X, y, depth=0, max_depth=5):
    # condition d'arret
    if len(np.unique(y)) == 1 or depth == max_depth:
        value = np.bincount(y).argmax()
        return Node(value=value)

    feature, threshold = best_split(X, y)

    if feature is None:
        value = np.bincount(y).argmax()
        return Node(value=value)

    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold

    left = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)

    return Node(feature, threshold, left, right)
#prediction
def predict_one(x, node):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_one(x, node.left)
    else:
        return predict_one(x, node.right)

def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

# Entraînement de l'arbre tree = build_tree(X_train, y_train, max_depth=5)
tree = build_tree(X_train, y_train, max_depth=5)
y_pred = predict(X_test, tree)

accuracy = (y_pred == y_test).mean()
print("Accuracy :", accuracy)

#les metrique matrice de confusion
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tn, fp, fn, tp

tn, fp, fn, tp = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")
print("TN:", tn, "FP:", fp)
print("FN:", fn, "TP:", tp)

#precision, rappel, f1-score
def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

p = precision(tp, fp)
r = recall(tp, fn)
f1 = f1_score(p, r)

print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1-score:  {f1:.3f}")


