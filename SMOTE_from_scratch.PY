import pickle
import numpy as np

def smote_binary_from_scratch(X, y, k=5, target_ratio=1.0, seed=42):

    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Cette fonction est pour classification binaire (2 classes).")

    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    n_min = counts[np.argmin(counts)]
    n_maj = counts[np.argmax(counts)]

    n_min_target = int(np.round(target_ratio * n_maj))
    if n_min_target <= n_min:
        return X, y

    n_to_generate = n_min_target - n_min

    X_min = X[y == minority_class]
    n_min_samples = X_min.shape[0]

    if n_min_samples < 2:
        raise ValueError("Pas assez d'échantillons minoritaires pour SMOTE.")

    k = min(k, n_min_samples - 1)

    norms = np.sum(X_min**2, axis=1, keepdims=True)
    dists = norms + norms.T - 2.0 * (X_min @ X_min.T)
    np.fill_diagonal(dists, np.inf)

    knn_idx = np.argsort(dists, axis=1)[:, :k]

    synth = np.empty((n_to_generate, X.shape[1]))

    for i in range(n_to_generate):
        base_i = rng.integers(0, n_min_samples)
        nn_i = knn_idx[base_i, rng.integers(0, k)]
        lam = rng.random()
        synth[i] = X_min[base_i] + lam * (X_min[nn_i] - X_min[base_i])

    X_res = np.vstack([X, synth])
    y_res = np.concatenate([y, np.full(n_to_generate, minority_class)])

    return X_res, y_res

# FONCTION run()

def run():
    print("\n=== Étape C : SMOTE from scratch ===")

    with open("data_preprocessed.pkl", "rb") as f:
        data = pickle.load(f)

    X_train = data["X_train"]
    y_train = data["y_train"]

    print("Avant SMOTE :", np.unique(y_train, return_counts=True))

    X_train_bal, y_train_bal = smote_binary_from_scratch(
        X_train, y_train, k=5, target_ratio=1.0, seed=42
    )

    print("Après SMOTE :", np.unique(y_train_bal, return_counts=True))

    data["X_train"] = X_train_bal
    data["y_train"] = y_train_bal

    with open("data_preprocessed.pkl", "wb") as f:
        pickle.dump(data, f)

    print("SMOTE appliqué et sauvegardé ")

