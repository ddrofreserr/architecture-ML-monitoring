import pandas as pd
import sqlite3


# def load_from_db() -> pd.DataFrame:
#     conn = sqlite3.connect("your_database.db")
#     df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
#     conn.close()
#     return df


def load_from_db(drop_target=False) -> pd.DataFrame:
    df = pd.read_csv("C:/Users/reserford/ded/practice/ds_course/yandex_practicum_projects/07 Выбор локации для скважины/geo_data_0.csv")

    df['target'] = df['stress'] // 100
    if drop_target:
        df = df.drop(['id', 'stress', 'target'], axis=1)
    else:
        df = df.drop(['id', 'stress'], axis=1)

    return df.head(20)
