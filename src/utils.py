
import logging
import sys
from pathlib import Path
from .config import LOG_FILE, LOGS_DIR

def setup_logging():
    """Configures logging for the application."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("customer_feedback_ai")

logger = setup_logging()
