from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from pydantic import BaseModel
from strictyaml import YAML, load

import classification_model

PACKAGE_ROOT = Path(classification_model.__file__).parent.resolve()
ROOT = PACKAGE_ROOT.parent

CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """Application-level config"""

    package_name: str
    raw_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """All configuration relevant to model training and feature engineering"""

    target: str
    unused_fields: Sequence[str]
    features: Sequence[str]
    test_size: float
    random_state: int
    numerical_vars: Sequence[str]


class Config(BaseModel):
    """Master config object"""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file"""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    if cfg_path is None:
        cfg_path = find_config_file()
    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path {cfg_path}")


def create_and_validate_config(
    parsed_config: Optional[Union[Dict[Any, Any], YAML]] = None
) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    if not isinstance(parsed_config, YAML):
        raise ValueError("Invalid parsed config")

    _config = Config(
        app_config=AppConfig(**parsed_config.data),  # ["app_config"]
        model_config=ModelConfig(**parsed_config.data),  # ["model_config"]
    )
    return _config


config = create_and_validate_config()
