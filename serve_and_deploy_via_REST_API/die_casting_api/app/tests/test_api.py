import math

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from classification_model.config.core import config


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given

    payload = {
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }
    #print(f'Функция test_make_prediction переменная payload : \n{payload["inputs"][:2]}')

    # Wnen
    response = client.post("http://localhost:8001/api/v1/predict", json=payload)
    prediction_data = response.json()
    print('Что возвращает client.post() смотрим response: \n')
    print(response.status_code)  # Статусный код ответа
    print(f"prediction_data['errors']: {prediction_data['errors']}\n")  # Тело ответа в формате JSON (если сервер возвращает JSON)
    print(f"prediction_data['version']: {prediction_data['version']}\n")  # Тело ответа в формате JSON (если сервер возвращает JSON)
    print(f"prediction_data['predictions']: {prediction_data['predictions']}")  # Тело ответа в формате JSON (если сервер возвращает JSON)

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["predictions"][5] == 0.0
    assert prediction_data["errors"] is None
    # assert math.isclose(prediction_data["predictions"][0], 113422, rel_tol=100)
