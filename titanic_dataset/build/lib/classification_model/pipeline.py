import sys

sys.path.append(
    "C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments/titanic_dataset/"
)

from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer

titanic_pipe = Pipeline(
    [
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars,
            ),
        ),
        (
            "missing_indicator",
            MeanMedianImputer(
                imputation_method="median", variables=config.model_config.numerical_vars
            ),
        ),
        (
            "extract_letter",
            ExtractLetterTransformer(variables=config.model_config.cabin_vars),
        ),
        (
            "rare_label_+encoder",
            RareLabelEncoder(
                tol=0.05, n_categories=1, variables=config.model_config.categorical_vars
            ),
        ),
        (
            "categorical_encoder",
            OneHotEncoder(
                drop_last=True, variables=config.model_config.categorical_vars
            ),
        ),
        ("scaler", StandardScaler()),
        ("Logit", LogisticRegression(C=0.0005, random_state=0)),
    ]
)
