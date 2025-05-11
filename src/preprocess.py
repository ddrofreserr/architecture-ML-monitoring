import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print(df.head())
    df = df.dropna(how="all")

    target_col = "target"

    features = df.drop(columns=[target_col], axis=1)
    target = df[target_col]

    f_train, f_val, t_train, t_val = train_test_split(features, target, test_size=0.25)
    return f_train, f_val, t_train, t_val
