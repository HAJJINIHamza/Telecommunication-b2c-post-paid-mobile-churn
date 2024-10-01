import logging
from datetime import datetime
import os

#THIS FILE ALLOWS US TO GENERATE LOGGINGS AND FOLLOWS OUR CODE

FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
logs_path = "artifacts/logs"
os.makedirs(logs_path, exist_ok=True)
FILE_PATH = os.path.join(logs_path, FILE_NAME)

logging.basicConfig(
    filename = FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level =logging.INFO
)