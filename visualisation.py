import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Heatmap optionnelle
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Importer uniquement des fonctions (decision_tree.py ne doit PAS charger de pickle à l'import)
from decision_tree import build_tree, predict

# Metrics from scratch
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (y_true == y_pred).mean()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return tn, fp, fn, tp, accuracy, precision, recall, f1

# Plots (robustes: save + close)
def plot_confusion_matrix(tn, fp, fn, tp, filename, title):
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(5, 4))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=["No", "Yes"],
                    yticklabels=["No", "Yes"])
    else:
        plt.imshow(cm)
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.xticks([0, 1], ["No", "Yes"])
        plt.yticks([0, 1], ["No", "Yes"])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_metrics_bar(acc, prec, rec, f1, filename, title):
    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy", "Precision", "Recall", "F1-score"],
            [acc, prec, rec, f1])
    plt.ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_loss_curve(loss_history, filename, title):
    train_loss = None
    if isinstance(loss_history, dict):
        train_loss = loss_history.get("train", None)

    if not train_loss:
        print("[WARN] loss_history vide, courbe loss ignorée.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(range(len(train_loss)), train_loss)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Log loss")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

# Causes de départ (depuis df_cleaned.csv)
def plot_overtime_attrition(df, out_dir):
    df2 = df.copy()
    df2["AttritionBin"] = df2["Attrition"].map({"No": 0, "Yes": 1})

    if "OverTime" not in df2.columns:
        print("[WARN] OverTime introuvable dans df_cleaned.csv")
        return

    rate = df2.groupby("OverTime")["AttritionBin"].mean() * 100

    plt.figure(figsize=(6, 4))
    plt.bar(rate.index.astype(str), rate.values)
    plt.ylabel("Taux d'attrition (%)")
    plt.title("Attrition selon OverTime")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cause_overtime.png"), dpi=200)
    plt.close()


def plot_jobrole_attrition(df, out_dir, top_n=10):
    df2 = df.copy()
    df2["AttritionBin"] = df2["Attrition"].map({"No": 0, "Yes": 1})

    if "JobRole" not in df2.columns:
        print("[WARN] JobRole introuvable dans df_cleaned.csv")
        return

    freq = df2["JobRole"].value_counts()
    keep = freq.head(top_n).index

    rate = df2.groupby("JobRole")["AttritionBin"].mean() * 100
    rate = rate.loc[keep].sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    plt.bar(rate.index.astype(str), rate.values)
    plt.ylabel("Taux d'attrition (%)")
    plt.title("Attrition par JobRole (Top catégories)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cause_jobrole.png"), dpi=200)
    plt.close()


def plot_monthly_income_boxplot(df, out_dir):
    if "MonthlyIncome" not in df.columns:
        print("[WARN] MonthlyIncome introuvable dans df_cleaned.csv")
        return

    income_no = df.loc[df["Attrition"] == "No", "MonthlyIncome"].dropna()
    income_yes = df.loc[df["Attrition"] == "Yes", "MonthlyIncome"].dropna()

    if income_no.empty or income_yes.empty:
        print("[WARN] Pas assez de données pour MonthlyIncome boxplot")
        return

    plt.figure(figsize=(6, 4))
    plt.boxplot([income_no, income_yes], labels=["Attrition=No", "Attrition=Yes"])
    plt.title("MonthlyIncome vs Attrition")
    plt.ylabel("MonthlyIncome")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cause_monthly_income_boxplot.png"), dpi=200)
    plt.close()

# Logistic Regression from scratch (minimal + stable)
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iter=1000, regularization="l2", lambda_reg=0.01, verbose=False, seed=42):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.reg = regularization
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.bias = 0.0
        self.loss_history = {"train": []}

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _loss(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        if self.reg == "l2":
            loss += (self.lambda_reg / 2) * np.sum(self.weights ** 2)
        elif self.reg == "l1":
            loss += self.lambda_reg * np.sum(np.abs(self.weights))
        return loss

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.weights = self.rng.normal(scale=0.01, size=n_features)
        self.bias = 0.0

        for i in range(self.n_iter):
            linear = X @ self.weights + self.bias
            y_pred = self._sigmoid(linear)

            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            if self.reg == "l2":
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.reg == "l1":
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = self._loss(y, y_pred)
            self.loss_history["train"].append(loss)

            if self.verbose and i % 200 == 0:
                print(f"[LR] iter {i}, loss={loss:.4f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# RUN
def run():
    out_dir = "images"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Charger les données ML
    with open("data_preprocessed.pkl", "rb") as f:
        data = pickle.load(f)

    X_train = data["X_train"]
    y_train = np.asarray(data["y_train"]).astype(int)
    X_test = data["X_test"]
    y_test = np.asarray(data["y_test"]).astype(int)

    # 2) Decision Tree
    print("\n=== Decision Tree (from scratch) ===")
    tree = build_tree(X_train, y_train, max_depth=5)
    y_pred_dt = predict(X_test, tree)

    tn, fp, fn, tp, acc, prec, rec, f1 = compute_metrics(y_test, y_pred_dt)
    print(f"[DT] Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

    plot_confusion_matrix(tn, fp, fn, tp,
                          filename=os.path.join(out_dir, "dt_confusion_matrix.png"),
                          title="Confusion Matrix - Decision Tree")
    plot_metrics_bar(acc, prec, rec, f1,
                     filename=os.path.join(out_dir, "dt_metrics_bar.png"),
                     title="Metrics - Decision Tree")

    # 3) Logistic Regression
    print("\n=== Logistic Regression (from scratch) ===")
    lr = LogisticRegressionScratch(learning_rate=0.01, n_iter=1000, regularization="l2", lambda_reg=0.01, verbose=False)
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test, threshold=0.5)
    tn, fp, fn, tp, acc, prec, rec, f1 = compute_metrics(y_test, y_pred_lr)
    print(f"[LR] Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

    plot_confusion_matrix(tn, fp, fn, tp,
                          filename=os.path.join(out_dir, "lr_confusion_matrix.png"),
                          title="Confusion Matrix - Logistic Regression")
    plot_metrics_bar(acc, prec, rec, f1,
                     filename=os.path.join(out_dir, "lr_metrics_bar.png"),
                     title="Metrics - Logistic Regression")
    plot_loss_curve(lr.loss_history,
                    filename=os.path.join(out_dir, "lr_loss_curve.png"),
                    title="Loss Curve - Logistic Regression")

    # 4) Causes de départ (depuis df_cleaned.csv)
    print("\n=== Causes de départ (df_cleaned.csv) ===")
    if os.path.exists("df_cleaned.csv"):
        df_clean = pd.read_csv("df_cleaned.csv")
        plot_overtime_attrition(df_clean, out_dir)
        plot_jobrole_attrition(df_clean, out_dir, top_n=10)
        plot_monthly_income_boxplot(df_clean, out_dir)
        print("[OK] Graphes causes sauvegardes dans images/")
    else:
        print("[WARN] df_cleaned.csv introuvable -> graphes causes ignorés.")

    print("\n Termine. Toutes les images sont dans le dossier 'images/'")


if __name__ == "__main__":
    run()

