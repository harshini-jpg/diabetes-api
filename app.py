

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

feature_cols = joblib.load('feature_cols.pkl')

model = joblib.load('catboost_diabetes.pkl')
app = FastAPI()

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get('/')

def home():
    return {
        'message':'Diabetes Prediction Running'
    }

@app.post('/predict')

def predict(data : DiabetesInput):
    input_df = pd.DataFrame([data])
    input_df = input_df[feature_cols]
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= 0.5)
    return {
        'prediction':prediction,
        'diabetes_probability':probability
    }
