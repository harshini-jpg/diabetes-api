
import joblib
import json
from config import (MODEL_PATH,FEATURE_PATH,METADATA_PATH)

feature_cols = joblib.load(FEATURE_PATH)

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH,'r') as f:
    model_metadata = json.load(f)

