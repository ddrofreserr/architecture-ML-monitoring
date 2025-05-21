import pandas as pd
from catboost import CatBoostClassifier
from .model_utils import explain_feature_influence
from sklearn.metrics import f1_score


def load_model():
    model = CatBoostClassifier()
    model.load_model("src/models/model.cbm")
    return model


def predict(df: pd.DataFrame) -> pd.Series:
    model = load_model()
    predictions = model.predict(df)
    return pd.Series(predictions, name="prediction")


def predict_with_explanation(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()

    threshold = 0.5 # Можно настроить на меньшее срабатывание
    
    # Получаем вероятности 
    proba = model.predict_proba(df)[:, 1]
    predictions = (proba >= threshold).astype(int) 

    explain_feature_influence(model, df, proba, threshold)
    
    result_df = df.copy()
    result_df['probability'] = proba
    result_df['prediction'] = predictions
    return result_df[['probability', 'prediction']]
