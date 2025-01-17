import sys

sys.path.append(
    "C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments/"
    "serve_and_deploy_via_REST_API/die_casting_api/"
)

from typing import Generator

import pandas as pd
import pytest
from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset, _load_raw_dataset
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    data = load_dataset(file_name=config.app_config.raw_data_file)

    _, X_test, _, _ = train_test_split(
        data[config.model_config.features],
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
