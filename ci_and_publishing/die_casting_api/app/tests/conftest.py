from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset, _load_raw_dataset
from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    data = _load_raw_dataset(file_name=config.app_config.raw_data_file)
    _, X_test, _, _ = train_test_split(
        data,
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    return X_test


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
