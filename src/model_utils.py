import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier, Pool

import matplotlib.pyplot as plt
import seaborn as sns


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Разделяет датасет на тренировочную и валидационную выборку и отделяет
    искомый параметр от тренировочных
    '''

    df = df.dropna(how="all")

    features = df.drop(columns=["target"], axis=1)
    target = df["target"]

    f_train, f_val, t_train, t_val = train_test_split(features, target, test_size=0.25)
    return f_train, f_val, t_train, t_val


def get_best_params(X, y, cat_features):
    '''
    Подбирает лучшие параметры для выбранной модели
    '''

    model = CatBoostClassifier(silent=True, cat_features=cat_features)

    param_dist = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'depth': list(range(4, 11, 2)),
        'l2_leaf_reg': [0, 1, 3, 5, 7, 9],
        'iterations': [300, 500, 1000],
    }

    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, random_state=42,
        n_iter=10, scoring='f1', cv=5, verbose=3, 
        # n_jobs=-1
    )

    random_search.fit(X, y)

    return random_search.best_params_


def explain_feature_influence(model: CatBoostClassifier, df: pd.DataFrame, proba: pd.Series, 
                              threshold: float, top_n: int = 3) -> None:
    '''
    На основе предсказанных данных отмечает, в каких местах обнаружены отклонения и какие параметры
    больше всего повлияли на предсказание
    '''

    predictions = (proba >= threshold).astype(int)
    shap_values = model.get_feature_importance(Pool(df), type="ShapValues")
    shap_values = shap_values[:, :-1]  # удаляем base value


    anomaly_indices = [idx for idx, pred in enumerate(predictions) if pred == 1]

    sorted_anomalies = sorted(
        [(idx, proba[idx]) for idx in anomaly_indices],
        key=lambda x: x[1],  # Сортируем по вероятности
        reverse=True
    )

    if not anomaly_indices:
        print("Отклонений в работе оборудования не зафиксировано.")
        return

    print("Обнаружены потенциальные отклонения в следующих наблюдениях:\n")

    for idx, prob in sorted_anomalies:
        row_id = df.iloc[idx].get('id', f"#{idx}")
        feature_influence = pd.Series(shap_values[idx], index=df.columns).abs().sort_values(ascending=False)
        top_features = feature_influence.head(top_n).index.tolist()
        top_values = feature_influence.head(top_n).values.tolist()

        top_str = ", ".join([f"{feat}={val:.2f}" for feat, val in zip(top_features, top_values)])
        print(f"— Объект ID {row_id}: вероятность аномалии {prob:.2f}, наибольшее влияние оказали признаки — {top_str}")


def plot_feature_importance(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series) -> None:

    '''
    Строит график важности признаков по обученной модели
    '''

    sns.set_style("whitegrid")

    # Важность по всем данным
    importances = model.get_feature_importance(type='PredictionValuesChange')
    importance_df = pd.DataFrame({'feature': list(X.columns), 'importance': importances})

    # Важность по target=0
    zero_pool = Pool(X[y == 0], label=y[y == 0])
    zero_importance = model.get_feature_importance(zero_pool, type='PredictionValuesChange')

    # Важность по target=1
    one_pool = Pool(X[y == 1], label=y[y == 1])
    one_importance = model.get_feature_importance(one_pool, type='PredictionValuesChange')

    importance_df['target_0_importance'] = zero_importance
    importance_df['target_1_importance'] = one_importance

    # Сортировка по общей важности
    importance_df.sort_values(by='importance', ascending=False, inplace=True)

    # Визуализация
    ind = np.arange(len(importance_df))
    width = 0.35  # ширина баров

    plt.figure(figsize=(10, 6))

    # Цвета для баров
    colors = sns.color_palette("Set2", n_colors=2)

    plt.barh(ind - width / 2, importance_df['target_0_importance'], width, label='Target = 0', color=colors[0])
    plt.barh(ind + width / 2, importance_df['target_1_importance'], width, label='Target = 1', color=colors[1])

    plt.yticks(ind, importance_df['feature'])
    plt.xlabel("Importance (PredictionValuesChange)")
    plt.title("Feature Importance для классов 0 и 1")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()
