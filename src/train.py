import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from .model_utils import preprocess, get_best_params, plot_feature_importance


def train_model(df: pd.DataFrame, retrain: bool = False) -> None:
    
    model_path = "./models/model.cbm"

    cat_features = list(df.select_dtypes(include=['object']).columns)

    X_train, X_val, y_train, y_val = preprocess(df)

    # дообучение существующей модели
    if retrain and os.path.exists(model_path):
        model = CatBoostClassifier()
        model.load_model(model_path)
        print("Продолжаем обучение существующей модели")

    # обучение новой модели
    else:
        # подбор лучших гиперпараметров
        best_params = get_best_params(X_train, y_train, cat_features)

        model = CatBoostClassifier(
            **best_params,
            cat_features=cat_features,
            verbose=100
        )
        print("Начинаем обучение новой модели")


    model.fit(
        X_train, y_train,
        init_model=model if retrain and os.path.exists(model_path) else None,
        use_best_model=False
    )

    # оценка точности на валидационной выборке
    print('F1 score:', f1_score(y_val, model.predict(X_val))) 

    # построение графиков влияния признаков
    plot_feature_importance(model, X_val, y_val)

    # сохранение модели
    os.makedirs("./models", exist_ok=True)
    model.save_model(model_path)
    print('Файл сохранен') if os.path.exists(model_path) else print('Файл не сохранен')
