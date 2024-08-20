import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import _load_raw_dataset, load_dataset
from classification_model.processing.features import Winsorizer


@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name=config.app_config.raw_data_file)
    _, X_test, _, _ = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )
    return X_test


def test_winsorizer_transformer(sample_input_data):
    # Given
    transformer = Winsorizer(variables=config.model_config.numerical_vars)

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    for var in config.model_config.numerical_vars:
        assert subject[var].min() >= sample_input_data[var].quantile(0.05)
        assert subject[var].max() <= sample_input_data[var].quantile(0.95)
