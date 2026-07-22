from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    model_config = {'extra': 'forbid'}

    fixed_acidity: float = Field(alias="fixed acidity", ge=3, le=16)
    volatile_acidity: float = Field(alias="volatile acidity", ge=0, le=2)
    citric_acid: float = Field(alias="citric acid", ge=0, le=2)
    residual_sugar: float = Field(alias="residual sugar", ge=0, le=30)
    chlorides: float = Field(ge=0, le=1)
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide", ge=0, le=300)
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide", ge=0, le=450)
    density: float = Field(ge=0.98, le=1.01)
    pH: float = Field(ge=2.5, le=4.5)
    sulphates: float = Field(ge=0, le=2.5)
    alcohol: float = Field(ge=7, le=16)
    color: Literal['red', 'white']

class PredictResponse(BaseModel):
    prediction: int
    prediction_float: float

class Wine(PredictRequest):
    name: str
    year: int | None = None
    country: str | None = None