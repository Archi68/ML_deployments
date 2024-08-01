import sys

sys.path.append(
    'C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments/'
    'titanic_dataset'
)

from typing import Any, List, Optional
from pydantic import BaseModel

from classification_model.processing.validation import TitanicDataInputsSchema

class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputsSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        'pclass': 1,
                        'survived': 1,
                        'name': 'Allen, Miss. Elisabeth Walton',
                        'sex': 'female',
                        'age': 29.0,
                        'sibsp': 0,
                        'parch': 0,
                        'ticket': '24160',
                        'fare': 211.3375,
                        'cabin': 'B5',
                        'embarked': 'S',
                        'boat': '2',
                        'body': 130,
                        'home.dest': 'St Louis, MO'
                    }
                ]
            }
        }
