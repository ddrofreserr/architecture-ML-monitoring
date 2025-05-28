import pandas as pd
import numpy as np


def generate_synthetic_data(n: int = 1000) -> pd.DataFrame:

    sensor_ids = [f"Sensor_{i:04d}" for i in range(1, n + 1)]
    materials = ['concrete'] * n

    data = {
        'Sensor ID': sensor_ids,
        'Material': materials,
        'Temperature': np.random.normal(loc=20, scale=5, size=n),
        'Humidity': np.random.beta(a=2, b=5, size=n) * 100, 
        'Vibration': np.random.lognormal(mean=0.5, sigma=0.3, size=n),
        'Crack Width':np.random.lognormal(mean=-3, sigma=0.3, size=n), 
        'Material Deformation': np.random.normal(loc=0.01, scale=0.002, size=n)
    }

    noise_std = 0.01
    for key in data:
        data[key] += np.random.normal(0, noise_std, size=n)

    return pd.DataFrame(data)
