
from config import LOG_FILE
import logging
logging.basicConfig(
    filename = LOG_FILE,
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
