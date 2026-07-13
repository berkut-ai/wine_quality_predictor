from schemas import PredictRequest, PredictResponse
import joblib
import json
import pandas as pd

with open('ml/models/config.json', 'r', encoding='utf-8') as conf_file:
    config = json.load(conf_file)

cat_boost = joblib.load('ml/models/catboost.pkl')
random_forest = joblib.load('ml/models/randomforest.pkl')

def predict(data: PredictRequest) -> PredictResponse:
    df = pd.DataFrame([data.model_dump(by_alias=True)])
    df['color'] = df['color'].map({'red': 0, 'white': 1})

    pred = (config['models']['catboost']['weight'] * cat_boost.predict(df) + config['models']['random_forest']['weight'] * random_forest.predict(df))[0]

    return PredictResponse(**{
        'prediction': round(pred),
        'prediction_float': round(pred, 3)
    })



if __name__ == '__main__':
    req = PredictRequest(**{
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
    })
    print(predict(req))
