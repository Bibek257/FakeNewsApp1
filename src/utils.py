import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging
import yaml
def save_object(file_path, obj):
    """
    Saves an object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}.")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads an object from a file using pickle.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}.")
        return obj
    except Exception as e:
        raise CustomException(e, sys)
    



def read_yaml(file_path: str):
    """Reads a YAML file and returns the contents as a dictionary.
    Args:
        file_path (str): The path to the YAML file.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    Raises: 
        FileNotFoundError: If the file is not found.
    """
    project_root = os.path.dirname(os.path.dirname(__file__))  
    abs_path = os.path.join(project_root, file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"YAML file not found at: {abs_path}")

    with open(abs_path, "r") as file:
        return yaml.safe_load(file)