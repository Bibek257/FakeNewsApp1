import logging
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
logs_folder = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_folder, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_folder, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s ] %(name)s %(levelname)s %(message)s',
    level=logging.INFO
)

if __name__ == "__main__":
 
    logging.info("Logging has been set up successfully.")
    print(f"Logs will be saved to {LOG_FILE_PATH}")