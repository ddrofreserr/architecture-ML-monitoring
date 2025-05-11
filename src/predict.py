import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

def predict(df: pd.DataFrame) -> pd.Series: # только данные с датчиков
    model = CatBoostClassifier()

    best_model = "src/models/model.cbm"

    model.load_model(best_model)

    predictions = model.predict(df)
    return pd.Series(predictions, name="prediction")


test = pd.read_csv("C:/Users/reserford/ded/practice/ds_course/yandex_practicum_projects/07 Выбор локации для скважины/geo_data_0.csv")
target = test['product'] // 100
features = test.drop(['id', 'product'], axis=1)

predictions = predict(features)
print(f1_score(target, predictions))
