import sys

sys.path.append(
    "C:\\Users\\USER\\OneDrive\\Docs\\KV_DW_JL_5\\ML_deployments\\"
    "serve_and_deploy_via_REST_API\\die_casting_api\\"
)

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

from classification_model import __version__ as model_version
from classification_model.predict import make_prediction
from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    health = schemas.Health(
        name=settings.PROJECT_NAME,
        api_version=__version__,
        model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDieCastingDataInputs) -> Any:
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making prediction on inputs: {input_data.inputs[:2]}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if not isinstance(results.get('predictions'), list):
        # Predictions must be a list
        results['predictions'] = list(results.get('predictions'))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")

        raise HTTPException(status_code=200, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results