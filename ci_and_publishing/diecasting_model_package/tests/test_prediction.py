import numpy as np

# import pandas as pd
import pytest

# from feature_engine.imputation import MeanMedianImputer
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.predict import make_prediction
from classification_model.processing.data_manager import (  # load_dataset,
    _load_raw_dataset,
    pre_pipeline_preparation,
    replace_nan_with_median,
)

# from classification_model.processing.validation import validate_inputs


@pytest.fixture
def sample_input_data():
    data = pre_pipeline_preparation(
        dataframe=_load_raw_dataset(file_name=config.app_config.raw_data_file)
    )

    _, X_test, _, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # Обработка отсутствующих значений в целевой переменной
    if y_test.isnull().any():
        y_test = replace_nan_with_median(y_test)

    return X_test, y_test


def test_make_prediction(sample_input_data):
    # Given
    X_test, y_test = sample_input_data
    expected_no_predictions = 1549  # len(X_test)

    # When
    result = make_prediction(input_data=X_test)

    # Then
    predictions = result.get("predictions")

    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], (np.float64, np.float32))
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions

    # print("y_test sample:", y_test[:10])
    # print("predictions sample:", predictions[:10])

    # Здесь используем функцию для замены NaN значений на медиану
    y_test = replace_nan_with_median(y_test)
    predictions = replace_nan_with_median(predictions)

    if len(np.unique(y_test)) > 1:
        try:
            y_test = y_test.astype(int)

            # Вычисляем precision, recall и AUC-PR
            precision, recall, _ = precision_recall_curve(y_test, predictions)
            auc_pr = auc(recall, precision)
            assert auc_pr > 0.6
        except ValueError as e:
            pytest.fail(f"ValueError during AUC-PR calculation: {e}")
    else:
        pytest.skip("Not enough classes in y_test to calculate AUC-PR score")