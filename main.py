import logging
from logging_config import setup_logging

setup_logging()

from fastapi import FastAPI, Body, HTTPException, Request
from inference import predict
from time import perf_counter
from typing import Annotated
from schemas import PredictResponse, PredictRequest, Wine
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get('/')
async def root():
    return {
        "status": "ok"
    }

@app.post('/v1/predict', response_model=PredictResponse)
@limiter.limit('30/minute')
async def create_predict(request: Request, data: Annotated[Wine, Body()]):
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