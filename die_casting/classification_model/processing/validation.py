import sys

sys.path.append("C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments/die_casting")

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config
from classification_model.processing.data_manager import pre_pipeline_preparation


class DieCastingDataInputSchema(BaseModel):
    Velocity_1: Optional[float]
    Velocity_2: Optional[float]
    Velocity_3: Optional[float]
    High_Velocity: Optional[float]
    Cylinder_Pressure: Optional[float]
    Rapid_Rise_Time: Optional[float]
    Biscuit_Thickness: Optional[float]
    Cycle_Time: Optional[float]
    Pressure_Rise_Time: Optional[float]
    Casting_Pressure: Optional[float]
    Spray_Time: Optional[float]
    Spray_1_Time: Optional[float]
    Spray_2_Time: Optional[float]


class MultipleDieCastingInputs(BaseModel):
    inputs: List[DieCastingDataInputSchema]


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    validated_data = input_data.copy()

    new_vars_with_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
    ]

    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    # pre_processed = pre_pipeline_preparation(dataframe=input_data)
    # validated_data = pre_processed[config.model_config.features].copy()

    pre_processed = pre_pipeline_preparation(dataframe=input_data)
    validated_data = drop_na_inputs(input_data=pre_processed)
    errors = None

    try:
        MultipleDieCastingInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors
