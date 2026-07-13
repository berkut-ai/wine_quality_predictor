from fastapi import FastAPI, Body
from inference import predict
from typing import Annotated
from schemas import PredictResponse, PredictRequest

app = FastAPI()

@app.post('/predict', response_model=PredictResponse)
async def create_predict(data: Annotated[PredictRequest, Body()]):
    return predict(data)