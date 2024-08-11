from typing import Any, List, Optional

from classification_model.processing.validation import DieCastingDataInputSchema
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleDieCastingDataInputs(BaseModel):
    inputs: List[DieCastingDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Velocity_1": 0.164,
                        "Velocity_2": 0.192,
                        "Velocity_3": 0.210,
                        "High_Velocity": 2.247,
                        "Cylinder_Pressure": 239.0,
                        "Rapid_Rise_Time": 0.009,
                        "Biscuit_Thickness": 13.0,
                        "Cycle_Time": 22.5,
                        "Pressure_Rise_Time": 0.039,
                        "Casting_Pressure": 1158.0,
                        "Spray_Time": 9.8,
                        "Spray_1_Time": 1.2,
                        "Spray_2_Time": 0.7,
                    }
                ]
            }
        }
