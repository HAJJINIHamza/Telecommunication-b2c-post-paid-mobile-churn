import os
from datetime import datetime
import logging

def setup_logger(class_name):
    # Create file name with timestamp and class name
    timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    file_name = f"{class_name}_{timestamp}.log"
    
    # Create the logs path if it doesn't exist
    logs_path = "artifacts/logs"
    os.makedirs(logs_path, exist_ok=True)
    file_path = os.path.join(logs_path, file_name)
    
    # Configure the logger
    logging.basicConfig(
        filename=file_path,
        format="[%(asctime)s] [%(class_name)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    # Add a custom logging filter to include class_name in the logs
    class ContextFilter(logging.Filter):
        def __init__(self, class_name):
            super().__init__()
            self.class_name = class_name

        def filter(self, record):
            record.class_name = self.class_name
            return True

    # Add the class_name filter to the logger
    logger = logging.getLogger()
    logger.addFilter(ContextFilter(class_name))

    return logger