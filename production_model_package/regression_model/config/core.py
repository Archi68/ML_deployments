'''import sys

sys.path.append("C:\\Users\\Irek9\\OneDrive\\Документы\\KV_DW_JL_5\\ML_deployments\\production_model_package")
'''
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load
import regression_model

PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent

ROOT = PACKAGE_ROOT.parent

CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"

DATASET_DIR = PACKAGE_ROOT / "datasets"

TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):

    target: str
    variables_to_rename: Dict
    features: List[str]
    test_size: float
    random_state: int
    alpha: float
    categorical_vars_with_na_frequent: List[str]
    categorical_vars_with_na_missing: List[str]
    numerical_vars_with_na: List[str]
    temporal_vars: List[str]
    ref_var: str
    numericals_log_vars: Sequence[str]
    binarize_vars: Sequence[str]
    qual_vars: List[str]
    exposure_vars: List[str]
    finish_vars: List[str]
    garage_vars: List[str]
    categorical_vars: Sequence[str]
    qual_mappings: Dict[str, int]
    exposure_mappings: Dict[str, int]
    garage_mappings: Dict[str, int]
    finish_mappings: Dict[str, int]


class Config (BaseModel):
    """
    Master config object
    """

    app_config: AppConfig

    model_config: ModelConfig

def find_config_file() -> Path:

    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:

    if not cfg_path:
        cfg_path = find_config_file()
    if cfg_path:
        with open(cfg_path, 'r') as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find at path {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:

    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Переменные начинающиеся с _ это временные переменные функции. Негласное соглашение питонистов
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        model_config = ModelConfig(**parsed_config.data)
    )

    return _config

config = create_and_validate_config()