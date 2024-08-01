'''import sys

sys.path.append('C:/Users/Irek9/OneDrive/Документы/KV_DW_JL_5/ML_deployments/production_model_package')
'''
import pytest

from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.test_data_file)

