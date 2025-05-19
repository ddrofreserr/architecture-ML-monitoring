import pandas as pd
import sqlite3


# def load_from_db() -> pd.DataFrame:
#     conn = sqlite3.connect("your_database.db")
#     df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
#     conn.close()
#     return df


def load_from_db() -> pd.DataFrame:
    df = pd.read_csv("C:/Users/reserford/ded/practice/ds_course/yandex_practicum_projects/07 Выбор локации для скважины/geo_data_0.csv")

    df['target'] = df['product'] // 100
    df = df.drop(['id', 'product'], axis=1)
    
    return df
