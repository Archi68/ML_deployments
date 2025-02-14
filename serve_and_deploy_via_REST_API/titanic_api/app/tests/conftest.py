import sys

sys.path.append(
    #'C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments/'
    #'titanic_dataset'
    'C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments'
    '/serve_and_deploy_via_REST_API/titanic_api/'
)

from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return load_dataset(file_name=config.app_config.test_data_file)


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
