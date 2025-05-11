import pandas as pd
from data.create_data import generate_synthetic_data
from data.db_utils import load_from_db

def load_data(source: str) -> pd.DataFrame:
    if source == "synthetic":
        return generate_synthetic_data()
    elif source == "db":
        return load_from_db()
