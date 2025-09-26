# DataPreparation/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

class DataPreprocessor:
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Cleaning data")
        return df.dropna()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Normalizing data")
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    def split(self, df: pd.DataFrame, target: str):
        logging.info("Splitting data into train and test sets")
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=0.2, random_state=42)