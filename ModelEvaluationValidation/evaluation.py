# ModelEvaluationValidation/evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import logging

class ModelEvaluator:
    def evaluate(self, y_true, y_pred):
        logging.info("Evaluating model")
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"MSE": mse, "R2": r2}

    def plot_residuals(self, y_true, y_pred, save_path="reports/residuals.png"):
        residuals = y_true - y_pred
        plt.figure()
        plt.scatter(y_pred, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.savefig(save_path)
        logging.info(f"Residual plot saved to {save_path}")