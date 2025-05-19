import pandas as pd
import numpy as np


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({
        "sensor_1": np.random.rand(n_samples),
        
        "sensor_2": np.random.rand(n_samples),
        "label": np.random.randint(0, 2, n_samples)
    })
    # тут точно добавить шумы
