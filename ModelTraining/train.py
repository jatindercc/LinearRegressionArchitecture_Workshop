# ModelTraining/train.py

import logging

class ModelTrainer:
    def train(self, model, X_train, y_train):
        logging.info("Training model")
        model.fit(X_train, y_train)
        return model