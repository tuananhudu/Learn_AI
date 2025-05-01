from math import * 
import numpy as np 

class RegressionMetrics:
    def __init__(self , y_pred , y_true):
        self.y_pred = y_pred 
        self.y_true = y_true 
    def mse(self):
        self.mse_value = np.mean((self.y_true - self.y_pred) **2)
        return self 
    def mae(self):
        self.mae_value = np.mean(abs(self.y_true - self.y_pred))
        return self 
    def rmse(self):
        self.rmse_value = sqrt(np.mean((self.y_true - self.y_pred) **2))
        return self 
    def r2_score(self):
        ss_total = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        self.r2_score_value =  1 - (ss_res / ss_total)
        return self 
    def summary(self):
        return {
            "mse": self.mse_value,
            "mae": self.mae_value,
            "rmse": self.rmse_value,
            "r2_score": self.r2_score_value
        }
    