import pandas as pd
import os
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from src.preprocess import preprocess

def train_model(df: pd.DataFrame, retrain: bool = False) -> None:
    
    model_path = "./models/model.cbm"

    X_train, X_val, y_train, y_val = preprocess(df)

    # когда будут признаки (скорее всего не будет)
    # cat_features = X_train.select_dtypes(include=["названия колонок"]).columns.tolist()

    # дообучение существующей модели
    if retrain and os.path.exists(model_path):
        model = CatBoostClassifier()
        model.load_model(model_path)
        print("Продолжаем обучение существующей модели")

    # обучение новой модели
    else:
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            eval_metric="Accuracy",
            # cat_features=cat_features,
            verbose=100,
            random_seed=42
        )
        print("Начинаем обучение новой модели")


    model.fit(
        X_train, y_train,
        init_model=model if retrain and os.path.exists(model_path) else None,
        use_best_model=False
    )

    print('F1 score:', f1_score(y_val, model.predict(X_val))) # оценка точности
    # print(model.feature_importances_) # влияние признаков

    # сохранение модели
    os.makedirs("./models", exist_ok=True)
    model.save_model(model_path)
    print('Файл сохранен') if os.path.exists(model_path) else print('Файл не сохранен')
