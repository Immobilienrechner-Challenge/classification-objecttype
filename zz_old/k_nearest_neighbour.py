import numpy as np
from scipy.stats import mode

class KNearestNeighbour:
    def __init__(self, k):
        self.k = k
    
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_test):
        labels_pred = []
        for x_test in X_test:
            distances = []
            for x_train in self.X_train:
                distance = np.linalg.norm(x_train - x_test)
                distances.append(distance)
            distances = np.array(distances)
            k_nearest = np.argsort(distances)[:self.k]
            labels = self.y_train[k_nearest]
            most_common = mode(labels, keepdims=True)
            majority = most_common.mode[0]
            labels_pred.append(majority)
        return np.array(labels_pred)