from catboost import CatBoostRegressor
from sklearn.ensemble import  RandomForestRegressor
import joblib
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'train_dataset.csv'
MODELS_DIR = BASE_DIR / 'models'

df = pd.read_csv(DATA_PATH)

X = df.drop('quality', axis=1)
y = df['quality']

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

with (MODELS_DIR / 'models_config.json').open('w', encoding='utf-8') as conf_file:
    json.dump(config, conf_file)

joblib.dump(cb, MODELS_DIR / 'catboost.pkl')
joblib.dump(rf, MODELS_DIR / 'randomforest.pkl')
