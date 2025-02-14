import sys

sys.path.append(
    # "C:\\Users\\Irek9\\OneDrive\\Документы\\KV_DW_JL_5\\ML_deployments\\titanic_dataset\\"
    "C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments/titanic_dataset/"
)

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config
from classification_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    pre_processed = pre_pipeline_preparation(dataframe=input_data)

    validated_data = pre_processed[config.model_config.features].copy()

    errors = None

    try:
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicDataInputsSchema(BaseModel):
    pclass: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[int]
    sibsp: Optional[int]
    parch: Optional[int]
    ticket: Optional[int]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    boat: Optional[Union[str, int]]
    body: Optional[int]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputsSchema]
