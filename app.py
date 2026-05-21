

from fastapi import FastAPI
import joblib
import pandas as pd
import logging
from fastapi import HTTPException
import json
from utils.logger import logger

from schemas.input_schema import DiabetesInput


feature_cols = joblib.load('models/feature_cols.pkl')

model = joblib.load('models/catboost_diabetes.pkl')

with open('models/model_metadata.json','r') as f:
    model_metadata = json.load(f)
    
app = FastAPI()

def model_info():
    logger.info("Model metadata requested")
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
        
