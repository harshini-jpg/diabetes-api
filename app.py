


from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi import HTTPException
import json
from utils.logger import logger

from schemas.input_schema import DiabetesInput

from config import MODEL_PATH,FEATURE_PATH,METADATA_PATH,PREDICTION_THRESHOLD
from utils.model_loader import model,feature_cols
    
app = FastAPI()

@app.get('/')
def home():

    return {
        'message': 'Diabetes Prediction Running'
    }


@app.get('/model-info')
def model_info():

    logger.info("Model metadata requested")

    return model_metadata
    
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
    
        prediction = int(probability >= PREDICTION_THRESHOLD)
    
        logger.info(
            f"Input : {input_data} |"
            f"Prediction : {prediction} |"
            f"Probability :{probability :0.4f}"
        )
    
        return {
            'prediction': prediction,
            'diabetes_probability': float(probability)
        }
    except Exception as e:
        logger.error(
            f"Prediction failed |"
            f"Input :{data.model_dump()} |"
            f"Error :{str(e)}"
        )
        raise HTTPException(
            status_code = 500,
            detail = str(e)
        )
        
