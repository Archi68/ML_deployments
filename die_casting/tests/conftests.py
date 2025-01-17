import sys

sys.path.append(
    "//Users//irinakim//Documents//ML_Deployment//die_casting//"
    "C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments/die_casting/"
)

import logging

import pytest
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name=config.app_config.raw_data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )
    return X_test
