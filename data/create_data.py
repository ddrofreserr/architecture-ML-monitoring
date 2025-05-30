import pandas as pd
import numpy as np


def vibration(temp):
    return np.log1p(np.abs(temp)) + np.random.normal(0, 0.1)


def generate_synthetic_data(n: int = 1000) -> pd.DataFrame:

    sensor_ids = [f"Sensor_{i:04d}" for i in range(1, n + 1)]
    materials = ['concrete'] * n

    days = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    t = np.arange(len(days))
    temperature = 10 + 15 * np.sin(2 * np.pi * t / (24 * 365)) + np.random.normal(0, 5, len(t))
    
    humidity = 100 - (temperature - 10) * 2 + np.random.normal(0, 5, n)
    vibrations = [vibration(t) for t in temperature]

    data = {
        'Sensor ID': sensor_ids,
        'Material': materials,
        'Temperature': temperature,
        'Humidity': humidity,
        'Vibration': vibrations,
        # 'Crack Width':np.random.lognormal(mean=-3, sigma=0.3, size=n), 
        'Material Deformation': np.random.normal(loc=0.01, scale=0.002, size=n)
    }

    noise_std = 0.01
    for key in data:
        data[key] += np.random.normal(0, noise_std, size=n)

    return pd.DataFrame(data)
