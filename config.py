
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH')
FEATURE_PATH = os.getenv('FEATURE_PATH')
METADATA_PATH = os.getenv('METADATA_PATH')
LOG_FILE = os.getenv('LOG_FILE')
PREDICTION_THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD',0.5))
