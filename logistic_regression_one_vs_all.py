import pandas as pd
import numpy as np


class LogisticRegressionOneVsAll:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.n_samples = None
        self.n_features = None
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_linear = self.linear_model(X)
            y_pred = np.array([1 if i > 0.5 else -1 for i in self.sigmoid(y_linear)])
            dw, db = self.gradient(X, y, y_pred)
            self.update_params(dw, db)


    def predict(self, X):
        y_linear = self.linear_model(X)
        y_pred = self.sigmoid(y_linear)
        return [1 if i > 0.5 else -1 for i in y_pred]


    def linear_model(self, X):
        return X @ self.weights + self.bias


    def sigmoid(self, y_pred):
        return 1 / (1 + np.exp(-y_pred))


    def gradient(self, X, y, y_pred):
        dw = (1 / self.n_samples) * X.T @ (y_pred - y)
        db = (1 / self.n_samples) * np.sum(y_pred - y)
        return dw, db

    
    def update_params(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
    
