import pandas as pd
import numpy as np


class LogisticClassification:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.n_samples = None
        self.n_features = None
        self.weights = None
        self.bias = None
        self.cost_hist = []


    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)[:, np.newaxis] # shape: (n_features, 1) (29, 1)
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = self.sigmoid(X @ self.weights + self.bias)
            derivative_cost = self.derivative_loss(y, y_pred)
            dw, db = self.gradient(X, y_pred, derivative_cost)
            self.update_params(dw, db)
            self.cost_hist.append(self.cost(y, y_pred))
    

    def gradient(self, X, y_pred, derivative_cost):
        derivative_weights = X.T @ (derivative_cost * y_pred * (1 - y_pred)) / self.n_samples
        derivative_bias = np.mean(derivative_cost * y_pred * (1 - y_pred)) / self.n_samples
        return derivative_weights, derivative_bias


    def cost(self, y, y_pred):
        return -(1 / self.n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


    def derivative_loss(self, y, y_pred):
        return -y / y_pred + (1 - y)/(1 - y_pred)


    def update_params(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db


    def predict(self, X):
        y_pred = self.sigmoid(X @ self.weights + self.bias)
        return np.array([1 if i > 0.5 else 0 for i in y_pred])


    def predict_proba(self, X):
        y_pred = X @ self.weights + self.bias
        return y_pred

    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))