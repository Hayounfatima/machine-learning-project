import pickle
import numpy as np

with open("data_preprocessed.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
y_train = data["y_train"]
X_test  = data["X_test"]
y_test  = data["y_test"]

# Feature selection by low variance 
variances = X_train.var(axis=0)
threshold = 0.01  
keep = variances > threshold

print("Features avant:", X_train.shape[1])
print("Features apres:", keep.sum())

X_train = X_train[:, keep]
X_test  = X_test[:, keep]

class LogisticRegressionScratch:
    """Regression logistique complète avec regularisation"""
    
    def __init__(self, learning_rate=0.01, n_iter=1000, 
                 regularization='l2', lambda_reg=0.01, 
                 verbose=False, seed=42):
        
        self.lr = learning_rate
        self.n_iter = n_iter
        self.reg = regularization
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.seed = seed
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.rng = np.random.default_rng(seed)
        
    def _sigmoid(self, z):
        """Fonction sigmoïde avec stabilite numerique"""
        # Pour éviter l'overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _loss(self, y_true, y_pred):
        """Fonction de perte logistique + régularisation"""
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
    
        # Perte logistique
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Régularisation
        if self.reg == 'l2' and self.weights is not None:
            loss += (self.lambda_reg / 2) * np.sum(self.weights ** 2)
        elif self.reg == 'l1' and self.weights is not None:
            loss += self.lambda_reg * np.sum(np.abs(self.weights))
        
        return loss
    
    def fit(self, X, y, validation_data=None):
        """
        Entraînement du modèle
        
        Parameters:
        -----------
        X : array, features d'entraînement
        y : array, labels d'entraînement
        validation_data : tuple (X_val, y_val), données de validation
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        n_samples, n_features = X.shape
        
        # Initialisation des paramètres
        self.weights = self.rng.normal(scale=0.01, size=n_features)
        self.bias = 0.0
        
        # Historique
        train_loss_history = []
        val_loss_history = []
        
        # Gradient descent
        for i in range(self.n_iter):
            # Forward pass
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear)
            
            # Calcul du gradient
            error = y_pred - y
            
            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            # Ajout de la régularisation
            if self.reg == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.reg == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
            
            # Mise à jour des paramètres
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Calcul de la perte
            train_loss = self._loss(y, y_pred)
            train_loss_history.append(train_loss)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.predict_proba(X_val)
                val_loss = self._loss(y_val, y_val_pred)
                val_loss_history.append(val_loss)
            
            # Affichage
            if self.verbose and i % 100 == 0:
                msg = f"Iteration {i}: loss = {train_loss:.4f}"
                if validation_data is not None:
                    msg += f", val_loss = {val_loss:.4f}"
                print(msg)
        
        self.loss_history = {
            'train': train_loss_history,
            'val': val_loss_history if validation_data else None
        }
        
        return self
    
    def predict_proba(self, X):
        """Prédit les probabilités"""
        X = np.asarray(X, dtype=float)
        linear = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear)
    
    def predict(self, X, threshold=0.5):
        """Prédit les classes"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_params(self):
        """Retourne les paramètres du modèle"""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias,
            'learning_rate': self.lr,
            'n_iter': self.n_iter,
            'regularization': self.reg,
            'lambda_reg': self.lambda_reg
        }