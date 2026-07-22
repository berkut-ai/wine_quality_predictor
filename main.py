import logging
from logging_config import setup_logging

setup_logging()

from fastapi import FastAPI, Body, HTTPException
from inference import predict
from time import perf_counter
from typing import Annotated
from schemas import PredictResponse, PredictRequest, Wine

logger = logging.getLogger(__name__)

app = FastAPI()

@app.post('/predict', response_model=PredictResponse)
async def create_predict(data: Annotated[Wine, Body()]):
    wine = f"{data.name}"

    logger.info(f"Accepted request for wine: {wine}.")
    try:
        start = perf_counter()
        result = predict(data)
        end = perf_counter()
    except Exception as err:
        logger.exception(f"Failed predict for wine: {wine}.", exc_info=err)
        raise HTTPException(status_code=500, detail="Prediction failed")
    else:
        logger.info(f"Predicted value: {result.prediction} in {round(end-start, 3)} sec.")
        return result