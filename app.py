

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import logging
from fastapi import HTTPException
import json

logging.basicConfig(
    filename = 'predictions_log.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

feature_cols = joblib.load('models/feature_cols.pkl')

model = joblib.load('models/catboost_diabetes.pkl')

with open('models/model_metadata.json','r') as f:
    model_metadata = json.load(f)
    
app = FastAPI()

from pydantic import BaseModel, Field


class DiabetesInput(BaseModel):

    Pregnancies: int = Field(ge=0, le=20)

    Glucose: float = Field(
        gt=0,
        le=500,
        description="Blood glucose level"
    )

    BloodPressure: float = Field(
        gt=0,
        le=300
    )

    SkinThickness: float = Field(
        ge=0,
        le=100
    )

    Insulin: float = Field(
        ge=0,
        le=1000
    )

    BMI: float = Field(
        gt=0,
        le=100
    )

    DiabetesPedigreeFunction: float = Field(
        ge=0,
        le=5
    )

    Age: int = Field(
        ge=1,
        le=120
    )

@app.get('/')

def model_info():
    logging.info("Model metadata requested")
    return model_metadata
    
def home():
    return {
        'message':'Diabetes Prediction Running'
    }

@app.post('/predict')

    
def predict(data: DiabetesInput):
    try:
        # Convert incoming request into dictionary
        input_data = data.model_dump()
    
        # Create dataframe
        input_df = pd.DataFrame([input_data])
    
        # Keep only training features in correct order
        input_df = input_df.reindex(columns=feature_cols)
    
        # Prediction
        probability = model.predict_proba(input_df)[0][1]
    
        prediction = int(probability >= 0.5)
    
        logging.info(
            f"Input : {input_data} |"
            f"Prediction : {prediction} |"
            f"Probability :{probability :0.4f}"
        )
    
        return {
            'prediction': prediction,
            'diabetes_probability': float(probability)
        }
    except Exception as e:
        logging.error(
            f"Prediction failed |"
            f"Input :{data.model_dump()} |"
            f"Error :{str(e)}"
        )
        raise HTTPException(
            status_code = 500,
            detail = 'Prediction Failed'
        )
        
