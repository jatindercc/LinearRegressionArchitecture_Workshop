# DataExtractionAnalysis/data_loader.py

import pandas as pd
import sqlite3
import requests
import logging

class DataExtractor:
    def load_csv(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading CSV from {path}")
        return pd.read_csv(path)

    def load_api(self, url: str) -> pd.DataFrame:
        logging.info(f"Fetching data from API: {url}")
        response = requests.get(url)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    def load_db(self, db_path: str, query: str) -> pd.DataFrame:
        logging.info(f"Querying database at {db_path}")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df