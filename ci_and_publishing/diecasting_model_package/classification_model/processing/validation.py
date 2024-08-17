from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config
from classification_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    validated_data = input_data[config.model_config.features].copy()

    errors = None

    try:
        MultipleDieCastingInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


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
