

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
def predict(data: DiabetesInput):

    # Convert incoming request into dictionary
    input_data = data.model_dump()

    # Create dataframe
    input_df = pd.DataFrame([input_data])

    # Keep only training features in correct order
    input_df = input_df.reindex(columns=feature_cols)

    # Prediction
    probability = model.predict_proba(input_df)[0][1]

    prediction = int(probability >= 0.5)

    return {
        'prediction': prediction,
        'diabetes_probability': float(probability)
    }
