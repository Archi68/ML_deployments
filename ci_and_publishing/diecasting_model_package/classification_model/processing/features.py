import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    """Преобразование для обрезания выбросов с использованием Winsorization"""

    def __init__(self, lower_percentile=0.05, upper_percentile=0.95, variables=None):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.variables = variables

    def fit(self, X, y=None):
        self.lower_bound_ = X[self.variables].quantile(self.lower_percentile)
        self.upper_bound_ = X[self.variables].quantile(self.upper_percentile)
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.clip(
                X[feature], self.lower_bound_[feature], self.upper_bound_[feature]
            )
        return X
