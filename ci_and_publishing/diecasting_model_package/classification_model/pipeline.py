from catboost import CatBoostClassifier
from feature_engine.imputation import MeanMedianImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from classification_model.config.core import config
from classification_model.processing.features import Winsorizer

die_casting_pipeline = Pipeline(
    steps=[
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median", variables=config.model_config.numerical_vars
            ),
        ),
        (
            "winsorizer",
            Winsorizer(variables=config.model_config.numerical_vars)
        ),
        (
            "scaler",
            MinMaxScaler()
        ),
        (
            "classifier",
            CatBoostClassifier(
                depth=9, iterations=185, learning_rate=0.13368726656560678
            ),
        ),
    ]
)
