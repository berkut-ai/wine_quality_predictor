from catboost import CatBoostRegressor
from sklearn.ensemble import  RandomForestRegressor
import joblib
import json
import pandas as pd

df = pd.read_csv('data/train_dataset.csv')

X, y = df.drop('quality', axis=1), df['quality']

cb = CatBoostRegressor(random_state=42, verbose=0)
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

cb.fit(X, y)
rf.fit(X, y)

config = {
  "models": {
    "catboost": {
      "path": "catboost.pkl",
      "weight": 0.29
    },
    "random_forest": {
      "path": "randomforest.pkl",
      "weight": 0.71
    }
  }
}

with open('models/config.json', 'w', encoding='utf-8') as conf_file:
    json.dump(config, conf_file)

joblib.dump(cb, 'models/catboost.pkl')
joblib.dump(rf, 'models/randomforest.pkl')
