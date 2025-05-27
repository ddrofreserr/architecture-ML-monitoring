import pandas as pd
import numpy as np


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({
        "temperature": np.random.rand(n_samples),
        "humidity": np.random.rand(n_samples),
        "target": np.random.randint(0, 2, n_samples)
    })
    # добавить шумы
