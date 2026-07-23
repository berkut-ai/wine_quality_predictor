# Wine Quality Predictor API

A REST API for predicting red and white wine quality using machine learning.

The service accepts wine physicochemical properties and returns a quality prediction on the scale used by the Wine Quality dataset. The API is built with **FastAPI** and uses a weighted ensemble of **CatBoost** and **Random Forest** models.

## Features

- Predicts the quality of red and white wine.
- Validates input types, required fields, and accepted value ranges.
- Includes interactive Swagger documentation out of the box.
- Logs requests and errors.
- Limits requests to **30 requests per minute per IP address**.
- Ships with serialized models, so no training is required to run the API.

## How predictions work

The project uses an ensemble of two models:

| Model | Weight in final prediction |
| --- | ---: |
| Random Forest | 71% |
| CatBoost | 29% |

Each response includes a rounded score and the exact prediction rounded to three decimal places.

## Project structure

```text
.
├── main.py                    # FastAPI application and /predict endpoint
├── inference.py               # Model loading and prediction logic
├── schemas.py                 # Pydantic request and response schemas
├── logging_config.py          # Logging configuration
├── requirements.txt           # Python dependencies
└── ml/
    ├── train.py               # Model training
    ├── data/                  # Training data
    └── models/                # Serialized models and configuration
```

## Quick start

Python 3.11+ is required.

```bash
git clone https://github.com/berkut-ai/wine_quality_predictor
cd wine_quality_predictor

python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use the following command instead of the last line:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and start the server:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API documentation

After starting the server, open one of the following pages in your browser:

| Page | URL |
| --- | --- |
| Swagger UI | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |
| ReDoc | [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) |
| OpenAPI JSON | [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json) |

### `POST /predict`

Accepts wine properties and returns a quality prediction.

```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Demo red wine",
    "year": 2018,
    "country": "Italy",
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "color": "red"
  }'
```

Example response:

```json
{
  "prediction": 5,
  "prediction_float": 5.123
}
```

`name` is required by the current API schema and is used in logs. `year` and `country` are optional and are not used to calculate the prediction.

### Request fields

| Field | Type | Constraint |
| --- | --- | --- |
| `fixed acidity` | number | 3–16 |
| `volatile acidity` | number | 0–2 |
| `citric acid` | number | 0–2 |
| `residual sugar` | number | 0–30 |
| `chlorides` | number | 0–1 |
| `free sulfur dioxide` | number | 0–300 |
| `total sulfur dioxide` | number | 0–450 |
| `density` | number | 0.98–1.01 |
| `pH` | number | 2.5–4.5 |
| `sulphates` | number | 0–2.5 |
| `alcohol` | number | 7–16 |
| `color` | string | `red` or `white` |

Invalid or extra fields are rejected with `422 Unprocessable Entity`.

## Rate limiting

The `/predict` endpoint is limited to 30 requests per minute per IP address. If the limit is exceeded, the service returns:

```text
429 Too Many Requests
```

The limiter stores counters in the application process memory. For multiple production instances, use a shared Redis store.
