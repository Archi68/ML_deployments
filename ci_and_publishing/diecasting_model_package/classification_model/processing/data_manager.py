import logging
from pathlib import Path
from typing import List, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

logger = logging.getLogger(__name__)


def replace_nan_with_median(arr):
    """Заменяет NaN значения в массиве на медиану не-NaN значений."""
    median = np.nanmedian(arr)  # Находим медиану, игнорируя NaN
    arr[np.isnan(arr)] = median  # Заменяем NaN на медиану

    return arr


def check_columns(data: pd.DataFrame, expected_columns: List[str]) -> None:
    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in dataset: {missing_columns}")


def pre_pipeline_preparation(*, dataframe: pd.DataFrame) -> pd.DataFrame:
    data = dataframe.replace("?", np.nan)

    # Проверка наличия дублирующихся строк и удаление при наличии
    if data.duplicated().sum() > 0:
        data.drop_duplicates(keep="last", inplace=True)

    """
    # Обработка отсутствующих значений в целевой переменной
    if data[config.model_config.target].isnull().any():
        _target = data[config.model_config.target]
        _target = replace_nan_with_median(_target)

        median = np.nanmedian(_target)  # Находим медиану, игнорируя NaN
        print(f'до _target[:5]: {_target[:5]}')
        _target.fillna(median)
        print(f'после _target[:5]: {_target[:5]}')
        #_target.iloc[np.isnan(_target)] = median  # Заменяем NaN на медиану
    """

    check_columns(data, list(config.model_config.features))
    # print(data.shape)

    return data


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(dataframe=dataframe)
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    return joblib.load(file_path)
