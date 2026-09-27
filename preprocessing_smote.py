import pandas as pd
import numpy as np
import pickle
 
def run(input_csv="df_cleaned.csv", output_pkl="data_preprocessed.pkl"):
    # 1) Charger le dataset nettoye
    df = pd.read_csv(input_csv)

    # 2) Separer X / y (Attrition)
    y = df["Attrition"].map({"No": 0, "Yes": 1}).to_numpy()
    X_df = df.drop(columns=["Attrition"])

    # 3) Encodage (ex: one-hot sur colonnes object)
    X_encoded = pd.get_dummies(X_df, drop_first=False)
    X = X_encoded.to_numpy(dtype=float)

    print("Features apres encodage :", X.shape[1])
    print("Dataset complet :", X.shape)

    # 4) Split stratifie 
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, seed=42)

    # 5) Standardisation appliquer sur TEST
    X_train_scaled, X_test_scaled, mean_, std_ = standardize_train_test(X_train, X_test)

    print("\nStandardisation terminee.")
    print("Train scaled:", X_train_scaled.shape, "Test scaled:", X_test_scaled.shape)

    # 6) Sauvegarder dans pickle
    output = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_names": X_encoded.columns.to_list(),
        "scaler_mean": mean_,
        "scaler_std": std_,
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(output, f)

    print(f"\n{output_pkl} sauvegarde.")

# ---------- Fonctions helpers ----------

def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    train_indices = []
    test_indices = []

    classes = np.unique(y)
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        n_test = int(np.ceil(test_size * len(idx_c)))
        test_c = idx_c[:n_test]
        train_c = idx_c[n_test:]

        test_indices.append(test_c)
        train_indices.append(train_c)

    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize_train_test(X_train, X_test, eps=1e-12):
    mean_ = X_train.mean(axis=0)
    std_ = X_train.std(axis=0)

    # eviter division par zero
    std_safe = np.where(std_ < eps, 1.0, std_)

    X_train_scaled = (X_train - mean_) / std_safe
    X_test_scaled  = (X_test - mean_) / std_safe

    return X_train_scaled, X_test_scaled, mean_, std_safe

if __name__ == "__main__":
    run()

  #separation train/test + standardisation + encodage
df = pd.read_csv("HR_Analytics.csv")

#  Encoder la variable cible (Attrition) Yes -> 1, No -> 0
if "Attrition" not in df.columns:
    raise ValueError("La colonne cible 'Attrition' est introuvable.")

df["Attrition"] = df["Attrition"].map({"No": 0, "Yes": 1})

if df["Attrition"].isnull().any():
    raise ValueError("Attrition contient des valeurs autres que Yes/No.")

#  Encoder les variables catégorielles 
#    On garde seulement le numérique après get_dummies
#  drop_first=True réduit la redondance (optionnel)
X_df = df.drop(columns=["Attrition"])
y = df["Attrition"].to_numpy()

X_encoded = pd.get_dummies(X_df, drop_first=True)
X = X_encoded.to_numpy(dtype=float)

print(" Features apres encodage :", X.shape[1])
print(" Dataset complet :", X.shape)

#  Split Train/Test STRATIFIÉ 
def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    """ 
    Split stratifié : conserve la proportion des classes dans train et test.
    - X: numpy array (n_samples, n_features)
    - y: numpy array (n_samples,)
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    train_indices = []
    test_indices = []

    classes = np.unique(y)
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        n_test = int(np.ceil(test_size * len(idx_c)))  # test proportion par classe
        test_c = idx_c[:n_test]
        train_c = idx_c[n_test:]

        test_indices.append(test_c)
        train_indices.append(train_c)

    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = stratified_train_test_split(
    X, y, test_size=0.2, seed=42
)

def class_distribution(y, name=""):
    values, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\n Distribution {name}:")
    for v, c in zip(values, counts):
        print(f"  Classe {v}: {c} ({(c/total)*100:.2f}%)")

class_distribution(y, "TOTAL")
class_distribution(y_train, "TRAIN")
class_distribution(y_test, "TEST")

#  Standardisation 
#    Fit sur train, transform sur test
class StandardScalerScratch:
    def __init__(self, eps=1e-12):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)  # ddof=0
        # éviter division par 0 si une feature est constante
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

scaler = StandardScalerScratch()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n Standardisation terminee.")
print("Train scaled:", X_train_scaled.shape, "Test scaled:", X_test_scaled.shape)

# Sauvegarde des sorties de l'étape test/train in + standardisation
#    ( appliquer SMOTE sur X_train_scaled + y_train)
output = {
    "X_train": X_train_scaled,
    "y_train": y_train,
    "X_test": X_test_scaled,
    "y_test": y_test,
    "feature_names": X_encoded.columns.to_list(),
    "scaler_mean": scaler.mean_,
    "scaler_std": scaler.std_,
}

with open("data_preprocessed.pkl", "wb") as f:
    pickle.dump(output, f)

print("\n data_preprocessed.pkl sauvegarde.")

'''
def run(input_csv="df_cleaned.csv", output_pkl="data_preprocessed.pkl"):
    # 1) Charger le dataset nettoyé
    df = pd.read_csv(input_csv)

    # 2) Séparer X / y (Attrition)
    y = df["Attrition"].map({"No": 0, "Yes": 1}).to_numpy()
    X_df = df.drop(columns=["Attrition"])

    # 3) Encodage (ex: one-hot sur colonnes object)
    X_encoded = pd.get_dummies(X_df, drop_first=False)
    X = X_encoded.to_numpy(dtype=float)

    print("Features apres encodage :", X.shape[1])
    print("Dataset complet :", X.shape)

    # 4) Split stratifié (from scratch)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, seed=42)

    # 5) Standardisation (from scratch) sur TRAIN, appliquer sur TEST
    X_train_scaled, X_test_scaled, mean_, std_ = standardize_train_test(X_train, X_test)

    print("\nStandardisation terminee.")
    print("Train scaled:", X_train_scaled.shape, "Test scaled:", X_test_scaled.shape)

    # 6) Sauvegarder dans pickle
    output = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_names": X_encoded.columns.to_list(),
        "scaler_mean": mean_,
        "scaler_std": std_,
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(output, f)

    print(f"\n{output_pkl} sauvegarde.")

# ---------- Fonctions helpers (from scratch) ----------

def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    train_indices = []
    test_indices = []

    classes = np.unique(y)
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        n_test = int(np.ceil(test_size * len(idx_c)))
        test_c = idx_c[:n_test]
        train_c = idx_c[n_test:]

        test_indices.append(test_c)
        train_indices.append(train_c)

    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize_train_test(X_train, X_test, eps=1e-12):
    mean_ = X_train.mean(axis=0)
    std_ = X_train.std(axis=0)

    # éviter division par zéro
    std_safe = np.where(std_ < eps, 1.0, std_)

    X_train_scaled = (X_train - mean_) / std_safe
    X_test_scaled  = (X_test - mean_) / std_safe

    return X_train_scaled, X_test_scaled, mean_, std_safe

if __name__ == "__main__":
    run()
'''