from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    model_config = {'extra': 'forbid'}

    fixed_acidity: float = Field(alias="fixed acidity")
    volatile_acidity: float = Field(alias="volatile acidity")
    citric_acid: float = Field(alias="citric acid")
    residual_sugar: float = Field(alias="residual sugar")
    chlorides: float
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide")
    density: float
    pH: float
    sulphates: float
    alcohol: float
    quality: int
    color: Literal['red', 'white']

class PredictResponse(BaseModel):
    prediction: int
    prediction_float: float