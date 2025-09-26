# ModelSelection/model.py

from sklearn.linear_model import LinearRegression
import logging

class ModelFactory:
    def create_model(self):
        logging.info("Creating Linear Regression model")
        return LinearRegression()