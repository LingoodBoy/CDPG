import sys
import logging
import sys
from datetime import datetime

def setup_logger(expIndex):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Generate timestamp for unique log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = "Logs/{}_{}_logs.log".format(expIndex, timestamp)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_filename)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger, log_filename