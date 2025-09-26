# ModelTraining/stream_simulator.py

import time
import logging
import pandas as pd
from sklearn.base import BaseEstimator

class StreamSimulator:
    def __init__(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, delay: float = 1.0):
        """
        Simulates streaming predictions one row at a time.
        :param model: Trained regression model.
        :param X_test: Test features.
        :param y_test: True labels.
        :param delay: Time delay between predictions (in seconds).
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.delay = delay

    def run(self):
        logging.info("Starting streaming simulation...")
        for i in range(len(self.X_test)):
            x = self.X_test.iloc[i:i+1]
            y_true = self.y_test.iloc[i]
            y_pred = self.model.predict(x)[0]
            print(f"[Stream] Row {i+1} → Actual: {y_true:.2f}, Predicted: {y_pred:.2f}")
            time.sleep(self.delay)
        logging.info("Streaming simulation completed.")